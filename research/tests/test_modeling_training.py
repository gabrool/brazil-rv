from __future__ import annotations

import copy
import csv
import gc
import importlib
import json
import platform
import sys
import tomllib
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch
import xgboost as xgb
from torch import nn

import brazil_rv.modeling.engine as engine_module
from brazil_rv.modeling.context_ablation import get_context_ablation
from brazil_rv.modeling.contract import (
    ADAMW_LR,
    ADAMW_WEIGHT_DECAY,
    BASELINE_TCN_SETTINGS,
    COMPILE_PARITY_EVALUATION_PREDICTION_MAX_ABSOLUTE,
    COMPILE_PARITY_EVALUATION_PREDICTION_RELATIVE_L2_MAX,
    COMPILE_PARITY_GRADIENT_COSINE_MIN,
    COMPILE_PARITY_GRADIENT_MAX_ABSOLUTE_ATOL,
    COMPILE_PARITY_GRADIENT_MAX_ABSOLUTE_RTOL,
    COMPILE_PARITY_GRADIENT_RELATIVE_L2_MAX,
    COMPILE_PARITY_LOSS_ATOL,
    COMPILE_PARITY_LOSS_RTOL,
    COMPILE_PARITY_PREDICTION_ATOL,
    COMPILE_PARITY_PREDICTION_RTOL,
    CompileEvaluationWarmupReport,
    CompileParityReport,
    CompileParityThresholds,
    CompileSetupReport,
    EFFECTIVE_BATCH_SIZE,
    EQUITY_COUNT,
    FINAL_LR_FACTOR,
    FEATURE_CONTRACT_VERSION,
    MAX_EPOCHS,
    NEURAL_MODELS,
    GH200_RUNTIME,
    HORIZONS,
    SANITY_MAX_LOSS,
    SANITY_MAX_LOSS_RATIO,
    SANITY_MIN_SPEARMAN,
    TABULAR_FEATURE_COUNT,
    TRANSFORMER_MODELS,
    TCNSettings,
    WARMUP_FRACTION,
    XGBOOST_CANDIDATES,
    XGBOOST_DEVICE,
    XGBOOST_FIXED_PARAMETERS,
    XGBOOST_VERSION,
    architecture_for_model,
    model_consumes_context,
    peer_feature_metadata,
)
from brazil_rv.modeling.evaluate import (
    _architecture_from_identity,
    _evaluation_context_ablation_fields,
    _normalize_context_ablation_identity,
    _validate_run_checkpoint_identity,
    _validate_xgboost_identity,
)
from brazil_rv.modeling.engine import (
    _compile_parity_report,
    _filter_evaluation_rows,
    build_compile_metadata,
    checkpoint_payload,
    clone_eager_reference_model,
    compile_model,
    objective_metadata,
    require_compile_parity,
    sam_metadata,
    validate_runtime,
)
from brazil_rv.modeling.metrics import create_metric_table
from brazil_rv.modeling.train import (
    _HISTORY_COLUMNS,
    _atomic_write_history,
    _atomic_write_json,
    _model_metadata,
)
from brazil_rv.modeling.model import build_neural_model
from brazil_rv.modeling.sanity import (
    _checkpoint_predictions_compatible,
    _memorization_passes,
    _sanity_checkpoint_payload,
)
from brazil_rv.modeling.data import TabularRowBatch
from brazil_rv.modeling.xgboost_model import (
    QuantileBatchDataIter,
    _fill_horizon_predictions,
    booster_device,
    build_quantile_matrix,
    inner_date_split,
    load_boosters,
    prediction_long_frame,
    qualify_native_cuda_xgboost,
    save_boosters,
    select_candidate,
    validate_booster_hashes,
)
from brazil_rv.modeling.optim import (
    build_optimizer,
    build_scheduler,
    learning_rate_factor,
    partition_parameters,
)


def _tcn_settings(model_name: str) -> TCNSettings | None:
    return BASELINE_TCN_SETTINGS if model_name == "tcn" else None


def _architecture(model_name: str):
    return architecture_for_model(model_name, _tcn_settings(model_name))


def _build_model(model_name: str) -> nn.Module:
    architecture = _architecture(model_name)
    return build_neural_model(
        model_name,
        architecture if model_name == "tcn" else None,
    )


def test_shared_prediction_routing_includes_peer_state_only_when_present() -> None:
    class InputCounter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inputs: tuple[torch.Tensor, ...] = ()

        def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
            self.inputs = inputs
            return torch.tensor(len(inputs))

    batch = {
        "patches": torch.tensor(1),
        "history_patch_mask": torch.tensor(2),
        "instrument_mask": torch.tensor(3),
        "slow_features": torch.tensor(4),
        "state_position": torch.tensor(5),
    }
    model = InputCounter()
    assert engine_module._predict(model, batch).item() == 5
    assert model.inputs == tuple(batch.values())

    peer_batch = {**batch, "peer_state": torch.tensor(6)}
    assert engine_module._predict(model, peer_batch).item() == 6
    assert model.inputs == tuple(peer_batch.values())


def test_compile_setup_modern_explicit_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_compile(module: nn.Module, **kwargs: object) -> None:
        calls.append(kwargs)

    config = SimpleNamespace(backward_pass_autocast="same_as_forward")
    monkeypatch.setattr(engine_module, "functorch_config", config)
    monkeypatch.setattr(nn.Module, "compile", fake_compile)
    runtime = GH200_RUNTIME
    report = compile_model(nn.Linear(2, 1), runtime)
    assert calls == [
        {
            "backend": runtime.compile_backend,
            "mode": runtime.compile_mode,
            "fullgraph": runtime.compile_fullgraph,
            "dynamic": runtime.compile_dynamic,
        }
    ]
    assert config.backward_pass_autocast == "off"
    assert report == CompileSetupReport(
        api="nn.Module.compile",
        backend="inductor",
        mode="default",
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
    runtime = GH200_RUNTIME
    report = compile_model(nn.Linear(2, 1), runtime)
    assert calls == [
        {
            "backend": runtime.compile_backend,
            "mode": runtime.compile_mode,
            "fullgraph": runtime.compile_fullgraph,
            "dynamic": runtime.compile_dynamic,
        }
    ]
    assert not hasattr(config, "backward_pass_autocast")
    assert not report.backward_pass_autocast_control_available
    assert report.backward_pass_autocast_policy == "legacy_implicit"


def test_compile_setup_requires_callable_module_api() -> None:
    model = SimpleNamespace(compile=None)
    with pytest.raises(RuntimeError, match="nn.Module.compile"):
        compile_model(model, GH200_RUNTIME)


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


def _assert_compile_parity_fails(report: CompileParityReport) -> None:
    assert not report.passed
    with pytest.raises(RuntimeError, match="Eager/compiled qualification failed"):
        require_compile_parity(report)


def test_compile_parity_exact_and_within_threshold_pass() -> None:
    assert _synthetic_compile_parity().passed
    eager_predictions = torch.tensor([[1.0, 2.0]])
    compiled_predictions = eager_predictions + 0.0078125
    report = _synthetic_compile_parity(
        prediction_pair=(eager_predictions, compiled_predictions),
        loss_pair=(1.0, 1.0001),
        gradient_pair=(
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.00001, 2.0]),
        ),
    )
    assert report.passed
    assert report.prediction_max_absolute_difference == pytest.approx(0.0078125)


def test_compile_parity_forward_only_accepts_observed_bf16_aggregate_error() -> None:
    eager_predictions = torch.tensor([[0.0, 8.05]])
    compiled_predictions = torch.tensor([[0.03125, 8.05]])
    report = _synthetic_compile_parity(
        prediction_pair=(eager_predictions, compiled_predictions),
        loss_pair=(0.9646787047386169, 0.9646696448326111),
        include_backward=False,
    )

    assert report.passed
    assert not report.prediction_allclose
    assert report.prediction_max_absolute_difference == pytest.approx(0.03125)
    assert report.prediction_relative_l2_error == pytest.approx(0.03125 / 8.05)
    assert report.evaluation_prediction_bounds_passed
    assert report.prediction_parity_passed
    require_compile_parity(report)


def test_compile_parity_observed_bf16_error_remains_strict_with_backward() -> None:
    eager_predictions = torch.tensor([[0.0, 8.05]])
    compiled_predictions = torch.tensor([[0.03125, 8.05]])
    report = _synthetic_compile_parity(
        prediction_pair=(eager_predictions, compiled_predictions),
        gradient_pair=(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0])),
    )

    assert not report.prediction_allclose
    assert report.evaluation_prediction_bounds_passed
    assert not report.prediction_parity_passed
    _assert_compile_parity_fails(report)


@pytest.mark.parametrize(
    ("relative_error", "expected_pass"),
    (
        (COMPILE_PARITY_EVALUATION_PREDICTION_RELATIVE_L2_MAX, True),
        (COMPILE_PARITY_EVALUATION_PREDICTION_RELATIVE_L2_MAX + 1e-5, False),
    ),
)
def test_compile_parity_forward_only_relative_l2_boundary(
    relative_error: float, expected_pass: bool
) -> None:
    eager_scale = 6.0
    eager_predictions = torch.tensor([[0.0, eager_scale]])
    compiled_predictions = torch.tensor([[eager_scale * relative_error, eager_scale]])
    report = _synthetic_compile_parity(
        prediction_pair=(eager_predictions, compiled_predictions),
        include_backward=False,
    )

    assert not report.prediction_allclose
    assert report.prediction_relative_l2_error == pytest.approx(relative_error)
    assert (
        report.prediction_max_absolute_difference
        < COMPILE_PARITY_EVALUATION_PREDICTION_MAX_ABSOLUTE
    )
    assert report.evaluation_prediction_bounds_passed is expected_pass
    assert report.prediction_parity_passed is expected_pass
    assert report.passed is expected_pass
    if not expected_pass:
        _assert_compile_parity_fails(report)


@pytest.mark.parametrize(
    ("maximum_error", "expected_pass"),
    (
        (COMPILE_PARITY_EVALUATION_PREDICTION_MAX_ABSOLUTE, True),
        (COMPILE_PARITY_EVALUATION_PREDICTION_MAX_ABSOLUTE + 1e-5, False),
    ),
)
def test_compile_parity_forward_only_maximum_absolute_boundary(
    maximum_error: float, expected_pass: bool
) -> None:
    eager_predictions = torch.tensor([[0.0, 10.0]])
    compiled_predictions = torch.tensor([[maximum_error, 10.0]])
    report = _synthetic_compile_parity(
        prediction_pair=(eager_predictions, compiled_predictions),
        include_backward=False,
    )

    assert not report.prediction_allclose
    assert report.prediction_max_absolute_difference == pytest.approx(maximum_error)
    assert (
        report.prediction_relative_l2_error
        < COMPILE_PARITY_EVALUATION_PREDICTION_RELATIVE_L2_MAX
    )
    assert report.evaluation_prediction_bounds_passed is expected_pass
    assert report.prediction_parity_passed is expected_pass
    assert report.passed is expected_pass
    if not expected_pass:
        _assert_compile_parity_fails(report)


@pytest.mark.parametrize("nonfinite_side", ("eager", "compiled"))
def test_compile_parity_forward_only_nonfinite_predictions_fail(
    nonfinite_side: str,
) -> None:
    eager_predictions = torch.tensor([[0.0, 8.05]])
    compiled_predictions = eager_predictions.clone()
    target = eager_predictions if nonfinite_side == "eager" else compiled_predictions
    target[0, 0] = float("nan")
    report = _synthetic_compile_parity(
        prediction_pair=(eager_predictions, compiled_predictions),
        include_backward=False,
    )

    assert not report.prediction_parity_passed
    _assert_compile_parity_fails(report)


def test_compile_parity_forward_only_loss_disagreement_still_fails() -> None:
    eager_predictions = torch.tensor([[0.0, 8.05]])
    compiled_predictions = torch.tensor([[0.03125, 8.05]])
    report = _synthetic_compile_parity(
        prediction_pair=(eager_predictions, compiled_predictions),
        loss_pair=(1.0, 1.006),
        include_backward=False,
    )

    assert report.prediction_parity_passed
    assert report.loss_absolute_difference > report.loss_tolerance
    _assert_compile_parity_fails(report)


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


def test_compile_parity_gradient_relative_l2_boundary() -> None:
    epsilon = 1e-4
    eager_gradient = torch.ones(100)
    inside = eager_gradient * (1.0 + COMPILE_PARITY_GRADIENT_RELATIVE_L2_MAX - epsilon)
    outside = eager_gradient * (1.0 + COMPILE_PARITY_GRADIENT_RELATIVE_L2_MAX + epsilon)
    inside_report = _synthetic_compile_parity(gradient_pair=(eager_gradient, inside))
    outside_report = _synthetic_compile_parity(gradient_pair=(eager_gradient, outside))
    assert inside_report.passed
    assert inside_report.gradient_relative_l2_error is not None
    assert (
        inside_report.gradient_relative_l2_error
        < COMPILE_PARITY_GRADIENT_RELATIVE_L2_MAX
    )
    assert not outside_report.passed
    assert outside_report.gradient_relative_l2_error is not None
    assert (
        outside_report.gradient_relative_l2_error
        > COMPILE_PARITY_GRADIENT_RELATIVE_L2_MAX
    )


def test_compile_parity_gradient_cosine_boundary() -> None:
    epsilon = 1e-5
    eager_gradient = torch.tensor([1.0, 0.0])

    def gradient(cosine: float) -> torch.Tensor:
        return torch.tensor([cosine, (1.0 - cosine**2) ** 0.5])

    inside_report = _synthetic_compile_parity(
        gradient_pair=(
            eager_gradient,
            gradient(COMPILE_PARITY_GRADIENT_COSINE_MIN + epsilon),
        )
    )
    outside_report = _synthetic_compile_parity(
        gradient_pair=(
            eager_gradient,
            gradient(COMPILE_PARITY_GRADIENT_COSINE_MIN - epsilon),
        )
    )
    assert inside_report.passed
    assert inside_report.gradient_cosine_similarity is not None
    assert inside_report.gradient_cosine_similarity > COMPILE_PARITY_GRADIENT_COSINE_MIN
    assert not outside_report.passed
    assert outside_report.gradient_cosine_similarity is not None
    assert (
        outside_report.gradient_cosine_similarity < COMPILE_PARITY_GRADIENT_COSINE_MIN
    )


def test_compile_parity_gradient_max_absolute_boundary() -> None:
    epsilon = 1e-4
    eager_gradient = torch.ones(100)
    tolerance = (
        COMPILE_PARITY_GRADIENT_MAX_ABSOLUTE_ATOL
        + COMPILE_PARITY_GRADIENT_MAX_ABSOLUTE_RTOL * float(eager_gradient.abs().max())
    )
    inside = eager_gradient.clone()
    outside = eager_gradient.clone()
    inside[0] += tolerance - epsilon
    outside[0] += tolerance + epsilon
    inside_report = _synthetic_compile_parity(gradient_pair=(eager_gradient, inside))
    outside_report = _synthetic_compile_parity(gradient_pair=(eager_gradient, outside))
    assert inside_report.passed
    assert inside_report.gradient_max_absolute_difference is not None
    assert inside_report.gradient_max_absolute_difference < tolerance
    assert not outside_report.passed
    assert outside_report.gradient_max_absolute_difference is not None
    assert outside_report.gradient_max_absolute_difference > tolerance


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
    assert report.prediction_allclose
    assert report.evaluation_prediction_bounds_passed
    assert report.prediction_parity_passed
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


@pytest.mark.parametrize(
    ("objective", "temperature"),
    (("soft_spearman", 0.1), ("rank_huber", None)),
)
def test_compile_qualification_uses_requested_grad_mode(
    monkeypatch: pytest.MonkeyPatch,
    objective: str,
    temperature: float | None,
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
            objective=objective,
            temperature=temperature,
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
            objective=objective,
            temperature=temperature,
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
        mode="default",
        fullgraph=True,
        dynamic=False,
        backward_pass_autocast_control_available=False,
        backward_pass_autocast_policy="legacy_implicit",
    )
    parity = _synthetic_compile_parity(
        prediction_pair=(
            torch.tensor([[0.0, 8.05]]),
            torch.tensor([[0.03125, 8.05]]),
        ),
        include_backward=False,
    )
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
    assert metadata["parity"]["prediction_allclose"] is False
    assert metadata["parity"]["evaluation_prediction_bounds_passed"] is True
    assert metadata["parity"]["prediction_parity_passed"] is True
    assert asdict(CompileParityThresholds()) == {
        "prediction_atol": COMPILE_PARITY_PREDICTION_ATOL,
        "prediction_rtol": COMPILE_PARITY_PREDICTION_RTOL,
        "evaluation_prediction_relative_l2_max": (
            COMPILE_PARITY_EVALUATION_PREDICTION_RELATIVE_L2_MAX
        ),
        "evaluation_prediction_max_absolute": (
            COMPILE_PARITY_EVALUATION_PREDICTION_MAX_ABSOLUTE
        ),
        "loss_atol": COMPILE_PARITY_LOSS_ATOL,
        "loss_rtol": COMPILE_PARITY_LOSS_RTOL,
        "gradient_relative_l2_max": COMPILE_PARITY_GRADIENT_RELATIVE_L2_MAX,
        "gradient_cosine_min": COMPILE_PARITY_GRADIENT_COSINE_MIN,
        "gradient_max_absolute_atol": COMPILE_PARITY_GRADIENT_MAX_ABSOLUTE_ATOL,
        "gradient_max_absolute_rtol": COMPILE_PARITY_GRADIENT_MAX_ABSOLUTE_RTOL,
    }


def test_sanity_checkpoint_prediction_compatibility_boundaries() -> None:
    eager = torch.tensor([1.0])
    tolerance = COMPILE_PARITY_PREDICTION_ATOL + COMPILE_PARITY_PREDICTION_RTOL
    inside = eager + tolerance - 1e-6
    outside = eager + tolerance + 1e-6
    assert _checkpoint_predictions_compatible(inside, eager)
    assert not _checkpoint_predictions_compatible(outside, eager)
    assert not _checkpoint_predictions_compatible(torch.tensor([float("nan")]), eager)
    assert not _checkpoint_predictions_compatible(eager, torch.tensor([float("inf")]))
    assert not _checkpoint_predictions_compatible(torch.ones(2), eager)


def test_sanity_memorization_gate_accepts_exact_boundaries() -> None:
    assert _memorization_passes(
        initial_loss=2.0 * SANITY_MAX_LOSS,
        final_loss=SANITY_MAX_LOSS,
        final_spearman=SANITY_MIN_SPEARMAN,
        predictions_finite=True,
        gradients_finite=True,
    )


@pytest.mark.parametrize(
    "overrides",
    (
        {"final_loss": SANITY_MAX_LOSS + 1e-6, "initial_loss": 1.0},
        {
            "final_loss": SANITY_MAX_LOSS_RATIO * 0.1 + 1e-6,
            "initial_loss": 0.1,
        },
        {"final_spearman": SANITY_MIN_SPEARMAN - 1e-6},
        {"predictions_finite": False},
        {"gradients_finite": False},
        {"initial_loss": 0.0, "final_loss": 0.0},
        {"initial_loss": float("nan")},
        {"final_loss": float("inf")},
        {"final_spearman": float("nan")},
    ),
)
def test_sanity_memorization_gate_rejects_each_failed_criterion(
    overrides: dict[str, float | bool],
) -> None:
    criteria: dict[str, float | bool] = {
        "initial_loss": 2.0 * SANITY_MAX_LOSS,
        "final_loss": SANITY_MAX_LOSS,
        "final_spearman": SANITY_MIN_SPEARMAN,
        "predictions_finite": True,
        "gradients_finite": True,
    }
    criteria.update(overrides)
    assert not _memorization_passes(**criteria)


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


def test_gh200_runtime_hardware_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = GH200_RUNTIME
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA GH200 480GB")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _: runtime.expected_compute_capability,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=runtime.minimum_vram_bytes),
    )
    monkeypatch.setattr(
        engine_module.system_platform,
        "machine",
        lambda: runtime.required_cpu_architecture,
    )
    monkeypatch.setattr(
        engine_module.system_platform, "platform", lambda: "test-platform"
    )
    hardware = validate_runtime()
    assert hardware.compute_capability == runtime.expected_compute_capability

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        validate_runtime()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError):
        validate_runtime()
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(RuntimeError):
        validate_runtime()
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=runtime.minimum_vram_bytes - 1),
    )
    with pytest.raises(RuntimeError):
        validate_runtime()
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=runtime.minimum_vram_bytes),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 1))
    with pytest.raises(RuntimeError):
        validate_runtime()
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _: runtime.expected_compute_capability,
    )
    monkeypatch.setattr(engine_module.system_platform, "machine", lambda: "x86_64")
    with pytest.raises(RuntimeError):
        validate_runtime()


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
        "sample_id": torch.tensor([100, 101, -1, -1]),
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
    np.testing.assert_array_equal(filtered["sample_id"], np.asarray([100, 101]))
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


def _feature_manifest(feature_store: Path) -> dict[str, object]:
    feature_store.mkdir(parents=True, exist_ok=True)
    source_hashes = {"ES": "source-sha256"}
    normalized_store_hashes = {"ES": "store-sha256"}
    manifest = {
        "contract_version": FEATURE_CONTRACT_VERSION,
        "global_context": {
            "source_hashes": source_hashes,
            "normalized_store_hashes": normalized_store_hashes,
        },
    }
    (feature_store / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "feature_manifest_contract_version": FEATURE_CONTRACT_VERSION,
        "global_context_source_hashes": source_hashes,
        "global_context_normalized_store_hashes": normalized_store_hashes,
        "context_ablation": get_context_ablation("none").metadata(),
    }


def _global_identity(feature_store: Path, model_name: str) -> dict[str, object]:
    identity = _feature_manifest(feature_store)
    identity["global_context"] = (
        "enabled"
        if model_consumes_context(model_name, _tcn_settings(model_name))
        else None
    )
    return identity


def _read_feature_manifest(feature_store: Path) -> dict[str, object]:
    _feature_manifest(feature_store)
    return json.loads((feature_store / "manifest.json").read_text(encoding="utf-8"))


def test_sanity_checkpoint_payload_records_canonical_provenance(
    tmp_path: Path,
) -> None:
    source_hashes = {
        "ES.v.0": "source-es-sha256",
        "NQ.v.0": "source-nq-sha256",
    }
    normalized_store_hashes = {
        "ES.v.0": "normalized-es-sha256",
        "NQ.v.0": "normalized-nq-sha256",
    }
    contract_version = "test-sanity-feature-contract"
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "contract_version": contract_version,
                "global_context": {
                    "source_hashes": source_hashes,
                    "normalized_store_hashes": normalized_store_hashes,
                },
            }
        ),
        encoding="utf-8",
    )
    model = _build_model("tcn")
    optimizer, _ = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    payload = _sanity_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        feature_store=tmp_path,
        git_commit_sha="sanity-test-sha",
    )

    assert payload["resolved_feature_store_path"] == str(tmp_path)
    assert payload["feature_manifest_contract_version"] == contract_version
    assert payload["global_context"] == "enabled"
    assert payload["global_context_source_hashes"] == source_hashes
    assert payload["global_context_normalized_store_hashes"] == normalized_store_hashes
    assert payload["git_commit_sha"] == "sanity-test-sha"


def _matching_run_identity(
    feature_store: Path,
    *,
    model_name: str = "tcn",
    optimizer_variant: str = "adamw",
    objective: str = "soft_spearman",
    temperature: float | None = 0.1,
    sam_rho: float | None = None,
    peer_features: str = "none",
) -> dict[str, object]:
    return {
        **_model_metadata(
            model_name,
            _architecture(model_name),
            _tcn_settings(model_name),
            peer_features,
        ),
        "optimizer_variant": optimizer_variant,
        "objective": objective_metadata(objective, temperature),
        "sam": sam_metadata(optimizer_variant, sam_rho),
        "seed": 11,
        **_global_identity(feature_store, model_name),
        "resolved_feature_store_path": str(feature_store),
        "git_commit_sha": "test-sha",
    }


def _json_round_tripped_tcn_identity(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object], Path]:
    feature_store = (tmp_path / "feature-store").resolve()
    checkpoint = _matching_run_identity(feature_store)
    manifest_path = tmp_path / "run_manifest.json"
    _atomic_write_json(manifest_path, checkpoint)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest, checkpoint, feature_store


def test_evaluation_identity_accepts_production_json_tuple_normalization(
    tmp_path: Path,
) -> None:
    manifest, checkpoint, feature_store = _json_round_tripped_tcn_identity(tmp_path)
    assert isinstance(checkpoint["architecture_constants"], dict)
    assert isinstance(manifest["architecture_constants"], dict)
    assert isinstance(checkpoint["architecture_constants"]["dilations"], tuple)
    assert isinstance(manifest["architecture_constants"]["dilations"], list)

    _validate_run_checkpoint_identity(manifest, checkpoint, feature_store)


def test_peer_identity_is_required_and_manifest_checkpoint_bound(
    tmp_path: Path,
) -> None:
    manifest, checkpoint, feature_store = _json_round_tripped_tcn_identity(tmp_path)
    missing = copy.deepcopy(manifest)
    del missing["peer_features"]
    with pytest.raises(
        ValueError, match="Missing run/checkpoint identity field: peer_features"
    ):
        _validate_run_checkpoint_identity(missing, checkpoint, feature_store)

    mismatch = copy.deepcopy(checkpoint)
    mismatch["peer_features"] = peer_feature_metadata(
        "tcn", _architecture("tcn"), "selected"
    )
    with pytest.raises(
        ValueError, match="Run/checkpoint identity mismatch: peer_features"
    ):
        _validate_run_checkpoint_identity(manifest, mismatch, feature_store)


def test_selected_peer_identity_reconstructs_mode_aware_parameter_count(
    tmp_path: Path,
) -> None:
    feature_store = (tmp_path / "feature-store").resolve()
    checkpoint = _matching_run_identity(feature_store, peer_features="selected")
    manifest_path = tmp_path / "run_manifest.json"
    _atomic_write_json(manifest_path, checkpoint)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_run_checkpoint_identity(manifest, checkpoint, feature_store)

    malformed = copy.deepcopy(manifest)
    malformed["peer_features"]["adapter"]["zero_initialized"] = False
    with pytest.raises(ValueError, match="Invalid peer-feature identity metadata"):
        _validate_run_checkpoint_identity(
            malformed, copy.deepcopy(malformed), feature_store
        )


@pytest.mark.parametrize("change", ("value", "sequence_order"))
def test_evaluation_identity_json_normalization_remains_strict(
    change: str, tmp_path: Path
) -> None:
    manifest, checkpoint, feature_store = _json_round_tripped_tcn_identity(tmp_path)
    constants = checkpoint["architecture_constants"]
    assert isinstance(constants, dict)
    if change == "value":
        constants["kernel_size"] = int(constants["kernel_size"]) + 1
    else:
        dilations = constants["dilations"]
        assert isinstance(dilations, tuple)
        constants["dilations"] = tuple(reversed(dilations))

    with pytest.raises(ValueError, match="architecture_constants"):
        _validate_run_checkpoint_identity(manifest, checkpoint, feature_store)


@pytest.mark.parametrize("missing_from", ("manifest", "checkpoint"))
def test_evaluation_identity_json_normalization_does_not_hide_missing_fields(
    missing_from: str, tmp_path: Path
) -> None:
    manifest, checkpoint, feature_store = _json_round_tripped_tcn_identity(tmp_path)
    identity = manifest if missing_from == "manifest" else checkpoint
    del identity["architecture_constants"]

    with pytest.raises(
        ValueError,
        match="Missing run/checkpoint identity field: architecture_constants",
    ):
        _validate_run_checkpoint_identity(manifest, checkpoint, feature_store)


def test_evaluation_identity_json_normalization_rejects_non_finite_numbers(
    tmp_path: Path,
) -> None:
    manifest = _matching_run_identity(tmp_path.resolve())
    checkpoint = copy.deepcopy(manifest)
    manifest["parameter_count"] = float("nan")
    checkpoint["parameter_count"] = float("nan")

    with pytest.raises(ValueError, match="not JSON-compatible: parameter_count"):
        _validate_run_checkpoint_identity(manifest, checkpoint, tmp_path.resolve())


@pytest.mark.parametrize("model_name", NEURAL_MODELS)
def test_evaluation_identity_accepts_every_model_architecture(
    model_name: str,
    tmp_path: Path,
) -> None:
    manifest = _matching_run_identity(tmp_path.resolve(), model_name=model_name)
    _validate_run_checkpoint_identity(manifest, dict(manifest), tmp_path.resolve())


@pytest.mark.parametrize(
    ("optimizer_variant", "sam_rho"),
    (("adamw", None), ("sam_adamw", 0.025)),
)
def test_evaluation_identity_accepts_optimizer_metadata(
    optimizer_variant: str,
    sam_rho: float | None,
    tmp_path: Path,
) -> None:
    manifest = _matching_run_identity(
        tmp_path.resolve(),
        optimizer_variant=optimizer_variant,
        sam_rho=sam_rho,
    )
    _validate_run_checkpoint_identity(manifest, dict(manifest), tmp_path.resolve())


@pytest.mark.parametrize(
    ("objective", "temperature"),
    (("soft_spearman", 0.1), ("rank_huber", None)),
)
def test_evaluation_identity_accepts_objective_metadata(
    objective: str,
    temperature: float | None,
    tmp_path: Path,
) -> None:
    manifest = _matching_run_identity(
        tmp_path.resolve(), objective=objective, temperature=temperature
    )
    _validate_run_checkpoint_identity(manifest, dict(manifest), tmp_path.resolve())


def test_legacy_neural_identity_synthesizes_canonical_none(tmp_path: Path) -> None:
    manifest = _matching_run_identity(tmp_path.resolve())
    checkpoint = copy.deepcopy(manifest)
    del manifest["context_ablation"]
    del checkpoint["context_ablation"]

    identity = _validate_run_checkpoint_identity(
        manifest, checkpoint, tmp_path.resolve()
    )

    assert identity.metadata == get_context_ablation("none").metadata()
    assert identity.source == "legacy_implicit_none"
    assert "context_ablation" not in manifest
    assert "context_ablation" not in checkpoint
    assert _evaluation_context_ablation_fields(identity) == {
        "context_ablation": get_context_ablation("none").metadata(),
        "context_ablation_identity_source": "legacy_implicit_none",
    }


def test_explicit_neural_identity_remains_strict_and_records_source(
    tmp_path: Path,
) -> None:
    manifest = _matching_run_identity(tmp_path.resolve())
    identity = _validate_run_checkpoint_identity(
        manifest, copy.deepcopy(manifest), tmp_path.resolve()
    )

    assert identity.metadata == get_context_ablation("none").metadata()
    assert identity.source == "explicit_registry_metadata"
    assert (
        _evaluation_context_ablation_fields(identity)[
            "context_ablation_identity_source"
        ]
        == "explicit_registry_metadata"
    )


@pytest.mark.parametrize("missing_from", ("manifest", "checkpoint"))
def test_neural_context_ablation_presence_mismatch_fails(
    missing_from: str, tmp_path: Path
) -> None:
    manifest = _matching_run_identity(tmp_path.resolve())
    checkpoint = copy.deepcopy(manifest)
    artifact = manifest if missing_from == "manifest" else checkpoint
    del artifact["context_ablation"]

    with pytest.raises(ValueError, match="context_ablation presence"):
        _validate_run_checkpoint_identity(manifest, checkpoint, tmp_path.resolve())


@pytest.mark.parametrize(
    ("metadata", "match"),
    (
        (None, "explicit object"),
        ({}, "valid key"),
        (
            {**get_context_ablation("none").metadata(), "key": "unknown"},
            "Unknown context ablation",
        ),
        (
            {
                **get_context_ablation("none").metadata(),
                "serialized_specification": "wrong",
            },
            "specification identity",
        ),
        (
            {
                **get_context_ablation("none").metadata(),
                "specification_sha256": "0" * 64,
            },
            "specification identity",
        ),
    ),
)
def test_explicit_neural_context_ablation_metadata_must_be_canonical(
    metadata: object, match: str, tmp_path: Path
) -> None:
    manifest = _matching_run_identity(tmp_path.resolve())
    checkpoint = copy.deepcopy(manifest)
    manifest["context_ablation"] = copy.deepcopy(metadata)
    checkpoint["context_ablation"] = copy.deepcopy(metadata)

    with pytest.raises(ValueError, match=match):
        _validate_run_checkpoint_identity(manifest, checkpoint, tmp_path.resolve())


def test_ablation_named_legacy_neural_run_cannot_fall_back_to_none(
    tmp_path: Path,
) -> None:
    manifest = _matching_run_identity(tmp_path.resolve())
    checkpoint = copy.deepcopy(manifest)
    del manifest["context_ablation"]
    del checkpoint["context_ablation"]
    run_dir = tmp_path / "tcn_ablation-drop_win_seed29_legacy"

    with pytest.raises(ValueError, match="Ablation-named"):
        _validate_run_checkpoint_identity(
            manifest, checkpoint, tmp_path.resolve(), run_dir
        )


def test_manifest_only_normalizer_rejects_explicit_noncanonical_metadata() -> None:
    manifest = {"context_ablation": get_context_ablation("drop_win").metadata()}
    identity = _normalize_context_ablation_identity(manifest)
    assert identity.metadata["key"] == "drop_win"
    malformed = {"context_ablation": {**identity.metadata, "description": "wrong"}}
    with pytest.raises(ValueError, match="specification identity"):
        _normalize_context_ablation_identity(malformed)


@pytest.mark.parametrize(
    "field",
    (
        "model_name",
        "optimizer_variant",
        "objective",
        "sam",
        "seed",
        "resolved_feature_store_path",
        "git_commit_sha",
        "tcn_settings",
        "architecture_constants",
        "parameter_count",
        "feature_manifest_contract_version",
        "global_context",
        "context_ablation",
        "global_context_source_hashes",
        "global_context_normalized_store_hashes",
    ),
)
def test_evaluation_identity_rejects_each_mismatch(field: str, tmp_path: Path) -> None:
    manifest = _matching_run_identity(tmp_path.resolve())
    checkpoint = copy.deepcopy(manifest)
    checkpoint[field] = {"mismatch": field}
    with pytest.raises(ValueError, match=field):
        _validate_run_checkpoint_identity(manifest, checkpoint, tmp_path.resolve())


@pytest.mark.parametrize(
    "field",
    (
        "feature_manifest_contract_version",
        "global_context_source_hashes",
        "global_context_normalized_store_hashes",
    ),
)
def test_evaluation_identity_rejects_feature_store_global_mismatch(
    field: str, tmp_path: Path
) -> None:
    manifest = _matching_run_identity(tmp_path.resolve())
    manifest[field] = {"mismatch": field}
    with pytest.raises(ValueError, match=field):
        _validate_run_checkpoint_identity(manifest, dict(manifest), tmp_path.resolve())


def test_evaluation_identity_rejects_internal_objective_and_optimizer_conflicts(
    tmp_path: Path,
) -> None:
    manifest = _matching_run_identity(tmp_path.resolve())
    manifest["objective"] = {**manifest["objective"], "temperature": 0.3}
    with pytest.raises(ValueError, match="temperature"):
        _validate_run_checkpoint_identity(
            manifest, copy.deepcopy(manifest), tmp_path.resolve()
        )

    manifest = _matching_run_identity(tmp_path.resolve())
    manifest["sam"] = sam_metadata("sam_adamw", 0.025)
    with pytest.raises(ValueError, match="SAM rho"):
        _validate_run_checkpoint_identity(
            manifest, copy.deepcopy(manifest), tmp_path.resolve()
        )


def test_evaluation_identity_rejects_incompatible_architecture_and_count(
    tmp_path: Path,
) -> None:
    manifest = _matching_run_identity(tmp_path.resolve())
    manifest["architecture_constants"] = {
        **manifest["architecture_constants"],
        "width": 127,
    }
    with pytest.raises(ValueError, match="architecture metadata"):
        _validate_run_checkpoint_identity(
            manifest, copy.deepcopy(manifest), tmp_path.resolve()
        )

    manifest = _matching_run_identity(tmp_path.resolve())
    manifest["parameter_count"] = 1
    with pytest.raises(ValueError, match="parameter count"):
        _validate_run_checkpoint_identity(
            manifest, copy.deepcopy(manifest), tmp_path.resolve()
        )

    manifest = _matching_run_identity(tmp_path.resolve())
    manifest["tcn_settings"] = {
        **manifest["tcn_settings"],
        "width": 64,
    }
    with pytest.raises(ValueError, match="architecture metadata"):
        _validate_run_checkpoint_identity(
            manifest, copy.deepcopy(manifest), tmp_path.resolve()
        )


def test_atomic_json_write_replaces_final_without_temporary_file(
    tmp_path: Path,
) -> None:

    output = tmp_path / "run_manifest.json"
    _atomic_write_json(output, {"status": "running"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "running"}
    assert not (tmp_path / "run_manifest.json.tmp").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("fusion", "none"),
        ("width", 64),
        ("receptive_field", "matched_full"),
        ("block", "silu"),
    ),
)
def test_evaluation_identity_rejects_each_tcn_setting_mismatch(
    field: str,
    value: object,
    tmp_path: Path,
) -> None:
    manifest = _matching_run_identity(tmp_path.resolve())
    manifest["tcn_settings"] = {
        **manifest["tcn_settings"],
        field: value,
    }
    with pytest.raises(ValueError, match="architecture metadata"):
        _validate_run_checkpoint_identity(
            manifest, copy.deepcopy(manifest), tmp_path.resolve()
        )


@pytest.mark.parametrize(
    ("settings_receptive_field", "architecture_receptive_field"),
    (("full", "matched_full"), ("matched_full", "full")),
)
def test_evaluation_identity_rejects_full_matched_full_crosswire(
    settings_receptive_field: str,
    architecture_receptive_field: str,
    tmp_path: Path,
) -> None:
    settings = TCNSettings("context_pooled", 128, settings_receptive_field, "gelu")
    architecture = architecture_for_model(
        "tcn",
        TCNSettings("context_pooled", 128, architecture_receptive_field, "gelu"),
    )
    identity = {
        **_model_metadata("tcn", architecture, settings),
        "optimizer_variant": "adamw",
        "objective": objective_metadata("soft_spearman", 0.1),
        "sam": sam_metadata("adamw", None),
        "seed": 11,
        **_feature_manifest(tmp_path),
        "global_context": "enabled",
        "resolved_feature_store_path": str(tmp_path),
        "git_commit_sha": "test-sha",
    }
    with pytest.raises(ValueError, match="architecture metadata"):
        _validate_run_checkpoint_identity(identity, dict(identity), tmp_path.resolve())


def _history_row(*, sam: bool, objective: str = "soft_spearman") -> dict[str, object]:
    return {
        "epoch": 3,
        "objective": objective,
        "soft_rank_temperature": 0.1 if objective == "soft_spearman" else None,
        "optimizer": "sam_adamw" if sam else "adamw",
        "rho": 0.025 if sam else None,
        "seed": 11,
        "optimizer_steps": 77,
        "backward_passes": 1_232 if sam else 616,
        "train_objective_loss": 0.8125,
        "validation_objective_loss": 0.75,
        "validation_primary_ic": 0.031,
        "validation_ic_30": 0.021,
        "validation_ic_60": 0.032,
        "validation_ic_120": 0.040,
        "mean_gradient_norm": 0.42,
        "maximum_gradient_norm": 0.91,
        "mean_first_pass_sam_gradient_norm": 0.33 if sam else None,
        "mean_sam_perturbation_norm": 0.025 if sam else None,
        "mean_second_pass_sam_gradient_norm": 0.37 if sam else None,
        "all_finite": True,
        "adamw_lr": 0.0003,
        "epoch_seconds": 61.5,
        "peak_allocated_cuda_memory_bytes": 12_345_678,
        "peak_reserved_cuda_memory_bytes": 23_456_789,
    }


@pytest.mark.parametrize("sam", (False, True), ids=("adamw", "sam_adamw"))
@pytest.mark.parametrize("objective", ("soft_spearman", "rank_huber"))
def test_atomic_history_round_trip_exact_schema_and_no_temporary_file(
    tmp_path: Path, sam: bool, objective: str
) -> None:
    output = tmp_path / "history.csv"
    output.write_text("stale\n", encoding="utf-8")
    expected = _history_row(sam=sam, objective=objective)

    _atomic_write_history(output, [expected])

    with output.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
    assert reader.fieldnames == list(_HISTORY_COLUMNS)
    assert rows == [
        {key: "" if value is None else str(value) for key, value in expected.items()}
    ]
    assert not output.with_name("history.csv.tmp").exists()


def test_atomic_history_rejects_unexpected_fields_without_replacing_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "history.csv"
    original = "existing-history\n"
    output.write_text(original, encoding="utf-8")
    row = {**_history_row(sam=False), "unexpected": "field"}

    with pytest.raises(ValueError, match="fields not in fieldnames"):
        _atomic_write_history(output, [row])

    assert output.read_text(encoding="utf-8") == original
    assert not output.with_name("history.csv.tmp").exists()


@pytest.mark.parametrize("model_name", NEURAL_MODELS)
def test_adamw_parameter_routing_is_complete_disjoint_and_semantic(
    model_name: str,
) -> None:
    model = _build_model(model_name)
    groups = partition_parameters(model)
    decay_ids = {id(parameter) for parameter in groups["decay"]}
    no_decay_ids = {id(parameter) for parameter in groups["no_decay"]}
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_ids = {id(parameter) for parameter in trainable}

    assert decay_ids.isdisjoint(no_decay_ids)
    assert decay_ids | no_decay_ids == trainable_ids
    assert len(groups["decay"]) + len(groups["no_decay"]) == len(trainable_ids)

    saw_linear_weight = False
    saw_bias = False
    saw_embedding = False
    saw_rms_norm = False
    for module in model.modules():
        for attribute, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            parameter_id = id(parameter)
            no_decay = (
                isinstance(module, (nn.RMSNorm, nn.Embedding))
                or attribute == "bias"
                or (module is model and attribute == "state_token")
            )
            assert (parameter_id in no_decay_ids) is no_decay
            assert (parameter_id in decay_ids) is (not no_decay)
            saw_linear_weight |= isinstance(module, nn.Linear) and attribute == "weight"
            saw_bias |= attribute == "bias"
            saw_embedding |= isinstance(module, nn.Embedding)
            saw_rms_norm |= isinstance(module, nn.RMSNorm)

    assert saw_linear_weight
    assert saw_bias
    assert saw_embedding is (model_name in TRANSFORMER_MODELS)
    assert hasattr(model, "state_token") is (model_name in TRANSFORMER_MODELS)
    if saw_rms_norm:
        assert all(
            id(module.weight) in no_decay_ids
            for module in model.modules()
            if isinstance(module, nn.RMSNorm)
        )

    optimizer, optimizer_groups = build_optimizer(model)
    assert optimizer_groups == groups
    assert {
        id(parameter) for parameter in optimizer.param_groups[0]["params"]
    } == decay_ids
    assert {
        id(parameter) for parameter in optimizer.param_groups[1]["params"]
    } == no_decay_ids
    assert optimizer.param_groups[0]["weight_decay"] == ADAMW_WEIGHT_DECAY
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_scheduler_warmup_cosine_endpoints_and_update_numbering() -> None:
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=ADAMW_LR)
    training_sample_count = 39_380
    scheduler, steps_per_epoch, warmup_steps = build_scheduler(
        optimizer, training_sample_count
    )
    total_steps = steps_per_epoch * MAX_EPOCHS

    assert steps_per_epoch == 77
    assert (
        steps_per_epoch
        == (training_sample_count + EFFECTIVE_BATCH_SIZE - 1) // EFFECTIVE_BATCH_SIZE
    )
    assert warmup_steps == int(WARMUP_FRACTION * total_steps)
    assert learning_rate_factor(1, total_steps, warmup_steps) == pytest.approx(
        1.0 / warmup_steps
    )
    assert learning_rate_factor(warmup_steps, total_steps, warmup_steps) == 1.0
    assert learning_rate_factor(
        total_steps, total_steps, warmup_steps
    ) == pytest.approx(FINAL_LR_FACTOR)

    observed = []
    for update_number in range(1, total_steps + 1):
        observed.append(optimizer.param_groups[0]["lr"] / ADAMW_LR)
        optimizer.step()
        scheduler.step()
        assert scheduler.last_epoch == update_number
    assert observed[0] == pytest.approx(
        learning_rate_factor(1, total_steps, warmup_steps)
    )
    assert observed[warmup_steps - 1] == pytest.approx(1.0)
    assert observed[-1] == pytest.approx(FINAL_LR_FACTOR)
    assert optimizer.param_groups[0]["lr"] / ADAMW_LR == pytest.approx(FINAL_LR_FACTOR)


@pytest.mark.parametrize("model_name", NEURAL_MODELS)
def test_checkpoint_round_trip_eager(model_name: str, tmp_path: Path) -> None:
    torch.manual_seed(13)
    model = _build_model(model_name)
    optimizer, _ = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    feature_manifest = _read_feature_manifest(tmp_path)
    global_context = (
        "enabled"
        if model_consumes_context(model_name, _tcn_settings(model_name))
        else None
    )
    payload = checkpoint_payload(
        model,
        optimizer,
        scheduler,
        model_name,
        _architecture(model_name),
        _tcn_settings(model_name),
        "adamw",
        "soft_spearman",
        0.1,
        None,
        11,
        1,
        0.0,
        tmp_path,
        global_context,
        feature_manifest,
        "test-sha",
    )
    manifest = {**_matching_run_identity(tmp_path, model_name=model_name)}
    _validate_run_checkpoint_identity(manifest, payload, tmp_path.resolve())
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored = _build_model(model_name)
    restored.load_state_dict(loaded["model_state_dict"])
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(
            restored.state_dict()[name], parameter, atol=0, rtol=0
        )


def test_peer_checkpoint_records_mode_representation_and_restores_adapter(
    tmp_path: Path,
) -> None:
    architecture = _architecture("tcn")
    model = build_neural_model("tcn", architecture, "selected")
    optimizer, _ = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    feature_manifest = _read_feature_manifest(tmp_path)
    payload = checkpoint_payload(
        model,
        optimizer,
        scheduler,
        "tcn",
        architecture,
        BASELINE_TCN_SETTINGS,
        "adamw",
        "soft_spearman",
        0.1,
        None,
        11,
        1,
        0.0,
        tmp_path,
        "enabled",
        feature_manifest,
        "test-sha",
        peer_features="selected",
    )
    assert payload["peer_features"]["mode"] == "selected"
    assert payload["peer_features"]["adapter"] == {
        "input_width": 10,
        "output_width": architecture.width,
        "bias": False,
        "zero_initialized": True,
        "injection_point": (
            "additive residual into equity states after temporal state construction "
            "and before existing TCN fusion and prediction head"
        ),
    }
    assert "peer_adapter.weight" in payload["model_state_dict"]
    restored = build_neural_model("tcn", architecture, "selected")
    restored.load_state_dict(payload["model_state_dict"])

    with pytest.raises(ValueError, match="Checkpoint peer mode does not match"):
        checkpoint_payload(
            model,
            optimizer,
            scheduler,
            "tcn",
            architecture,
            BASELINE_TCN_SETTINGS,
            "adamw",
            "soft_spearman",
            0.1,
            None,
            11,
            1,
            0.0,
            tmp_path,
            "enabled",
            feature_manifest,
            "test-sha",
        )


@pytest.mark.parametrize(
    "settings",
    (
        TCNSettings("pooled_market", 192, "long", "gelu"),
        TCNSettings("pooled_market", 192, "long", "silu"),
        TCNSettings("pooled_market", 192, "long", "swiglu"),
        TCNSettings("context_pooled", 128, "matched_full", "gelu"),
    ),
)
def test_selected_tcn_checkpoint_round_trip_and_identity(
    tmp_path: Path, settings: TCNSettings
) -> None:
    architecture = architecture_for_model("tcn", settings)
    torch.manual_seed(31)
    model = build_neural_model("tcn", architecture)
    optimizer, _ = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    feature_manifest = _read_feature_manifest(tmp_path)
    global_context = "enabled" if model_consumes_context("tcn", settings) else None
    payload = checkpoint_payload(
        model,
        optimizer,
        scheduler,
        "tcn",
        architecture,
        settings,
        "adamw",
        "soft_spearman",
        0.1,
        None,
        11,
        2,
        0.05,
        tmp_path,
        global_context,
        feature_manifest,
        "test-sha",
    )
    manifest = {
        **_model_metadata("tcn", architecture, settings),
        "optimizer_variant": "adamw",
        "objective": objective_metadata("soft_spearman", 0.1),
        "sam": sam_metadata("adamw", None),
        "seed": 11,
        "resolved_feature_store_path": str(tmp_path),
        **_feature_manifest(tmp_path),
        "global_context": global_context,
        "git_commit_sha": "test-sha",
    }
    _validate_run_checkpoint_identity(manifest, payload, tmp_path.resolve())
    assert payload["tcn_settings"] == asdict(settings)
    assert payload["architecture_constants"] == asdict(architecture)
    assert payload["parameter_count"] == manifest["parameter_count"]

    checkpoint_path = tmp_path / "selected.pt"
    torch.save(payload, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert _architecture_from_identity(loaded) == architecture
    restored = build_neural_model("tcn", architecture)
    restored.load_state_dict(loaded["model_state_dict"])
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(
            restored.state_dict()[name], parameter, atol=0.0, rtol=0.0
        )


def test_gh200_cupy_dependency_is_pinned_locked_and_importable() -> None:
    research_root = Path(__file__).parents[1]
    project = tomllib.loads((research_root / "pyproject.toml").read_text("utf-8"))
    requirement = (
        "cupy-cuda12x==14.1.1; "
        "sys_platform == 'linux' and platform_machine == 'aarch64'"
    )
    assert requirement in project["project"]["dependencies"]

    lock = tomllib.loads((research_root / "uv.lock").read_text("utf-8"))
    cupy_packages = [
        package for package in lock["package"] if package["name"] == "cupy-cuda12x"
    ]
    assert len(cupy_packages) == 1
    assert cupy_packages[0]["version"] == "14.1.1"
    root_package = next(
        package for package in lock["package"] if package["name"] == "brazil-rv"
    )
    marker = "platform_machine == 'aarch64' and sys_platform == 'linux'"
    assert {"name": "cupy-cuda12x", "marker": marker} in root_package["dependencies"]
    assert {
        "name": "cupy-cuda12x",
        "marker": marker,
        "specifier": "==14.1.1",
    } in root_package["metadata"]["requires-dist"]
    assert XGBOOST_VERSION == xgb.__version__ == "3.2.0"

    if sys.platform == "linux" and platform.machine() == "aarch64":
        cupy = importlib.import_module("cupy")
        assert cupy.__version__ == "14.1.1"


def test_xgboost_candidate_order_and_deterministic_tie_breaking() -> None:
    expected = [
        (depth, learning_rate, child_weight)
        for depth in (4, 6, 8)
        for learning_rate in (0.03, 0.07)
        for child_weight in (10, 50)
    ]
    assert [
        (row.max_depth, row.learning_rate, row.min_child_weight)
        for row in XGBOOST_CANDIDATES
    ] == expected
    rows = [
        {"candidate_index": index, "primary_score": 0.25}
        for index in range(len(XGBOOST_CANDIDATES))
    ]
    assert select_candidate(rows)["candidate_index"] == 0


def test_xgboost_inner_split_uses_final_twenty_percent_and_five_date_embargo() -> None:
    dates = pl.date_range(
        pl.date(2024, 1, 2),
        pl.date(2024, 2, 12),
        interval="1d",
        eager=True,
    )[:30]
    rows = pl.DataFrame(
        {
            "sample_id": np.arange(30),
            "trade_date": dates,
        }
    )
    split = inner_date_split(rows)
    assert len(split.training_dates) == 19
    assert len(split.embargo_dates) == 5
    assert len(split.validation_dates) == 6
    assert split.training_rows.height == 19
    assert split.validation_rows.height == 6
    assert max(split.training_dates) < min(split.embargo_dates)
    assert max(split.embargo_dates) < min(split.validation_dates)


def test_xgboost_prediction_reshape_and_long_frame() -> None:
    output = np.zeros((2, EQUITY_COUNT, len(HORIZONS)), dtype=np.float32)
    source = [
        TabularRowBatch(
            features=np.zeros((2, TABULAR_FEATURE_COUNT), dtype=np.float32),
            labels=np.zeros(2, dtype=np.float32),
            weights=np.ones(2, dtype=np.float32),
            sample_id=np.asarray([20, 10], dtype=np.int64),
            date_idx=np.asarray([2, 1], dtype=np.int64),
            decision_idx=np.asarray([5, 4], dtype=np.int64),
            equity_slot=np.asarray([3, 5], dtype=np.int64),
        )
    ]
    _fill_horizon_predictions(
        output,
        1,
        np.asarray([0.4, -0.2], dtype=np.float32),
        source,  # type: ignore[arg-type]
        {10: 0, 20: 1},
    )
    assert output.shape == (2, 158, 3)
    assert output[1, 3, 1] == pytest.approx(0.4)
    assert output[0, 5, 1] == pytest.approx(-0.2)

    mask = np.zeros_like(output, dtype=bool)
    mask[1, 3, 1] = True
    arrays = {
        "sample_id": np.asarray([10, 20], dtype=np.int64),
        "date_idx": np.asarray([1, 2], dtype=np.int64),
        "decision_idx": np.asarray([4, 5], dtype=np.int64),
        "label_mask": mask,
        "targets": output + 1.0,
        "raw_returns": output - 1.0,
    }
    rows = pl.DataFrame(
        {
            "sample_id": [10, 20],
            "date_idx": [1, 2],
            "decision_idx": [4, 5],
        }
    )
    frame = prediction_long_frame(rows, output, arrays)
    assert frame.to_dicts() == [
        {
            "sample_id": 20,
            "date_idx": 2,
            "decision_idx": 5,
            "equity_slot": 3,
            "horizon_minutes": 60,
            "prediction": pytest.approx(0.4),
            "target": pytest.approx(1.4),
            "raw_return": pytest.approx(-0.6),
        }
    ]


def _tabular_xgboost_test_batch() -> TabularRowBatch:
    return TabularRowBatch(
        features=np.arange(4 * TABULAR_FEATURE_COUNT, dtype=np.float32).reshape(
            4, TABULAR_FEATURE_COUNT
        ),
        labels=np.asarray([-1.0, -0.25, 0.25, 1.0], dtype=np.float32),
        weights=np.full(4, 0.25, dtype=np.float32),
        sample_id=np.arange(4, dtype=np.int64),
        date_idx=np.zeros(4, dtype=np.int64),
        decision_idx=np.zeros(4, dtype=np.int64),
        equity_slot=np.arange(4, dtype=np.int64),
    )


def test_xgboost_cpu_iterator_is_explicit_and_passes_numpy() -> None:
    captured: dict[str, object] = {}
    iterator = QuantileBatchDataIter([_tabular_xgboost_test_batch()], device="cpu")
    assert iterator.next(lambda **values: captured.update(values))
    assert isinstance(captured["data"], np.ndarray)
    assert isinstance(captured["label"], np.ndarray)
    assert isinstance(captured["weight"], np.ndarray)
    assert not iterator.next(lambda **_: None)


def test_xgboost_cuda_iterator_rejects_numpy_batches() -> None:
    iterator = QuantileBatchDataIter([_tabular_xgboost_test_batch()], device="cuda")
    with pytest.raises(TypeError, match="CUDA torch.Tensor"):
        iterator.next(lambda **_: None)


def test_xgboost_streaming_external_memory_quantile_matrix(tmp_path: Path) -> None:
    matrix = build_quantile_matrix(
        [_tabular_xgboost_test_batch()],
        tmp_path / "quantile",
        device="cpu",
    )
    assert matrix.num_row() == 4
    assert matrix.num_col() == TABULAR_FEATURE_COUNT
    assert any(path.is_file() for path in tmp_path.iterdir())
    del matrix
    gc.collect()
    assert not any(path.is_file() for path in tmp_path.rglob("*"))


def _train_cpu_xgboost_test_model() -> tuple[xgb.Booster, xgb.DMatrix]:
    features = np.arange(48, dtype=np.float32).reshape(12, 4)
    labels = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
    matrix = xgb.DMatrix(features, label=labels)
    booster = xgb.train(
        {
            "objective": "reg:pseudohubererror",
            "huber_slope": 1.0,
            "tree_method": "hist",
            "device": "cpu",
            "seed": 17,
        },
        matrix,
        num_boost_round=3,
    )
    return booster, matrix


def test_xgboost_three_booster_ubj_round_trip(tmp_path: Path) -> None:
    booster, matrix = _train_cpu_xgboost_test_model()
    boosters = {index: booster for index in range(len(HORIZONS))}
    expected = booster.predict(matrix)
    hashes = save_boosters(tmp_path, boosters)
    assert set(hashes) == {"30m", "60m", "120m"}
    assert all(len(digest) == 64 for digest in hashes.values())

    loaded = load_boosters(tmp_path, requested_device="cpu", expected_sha256=hashes)
    for index, horizon in enumerate(HORIZONS):
        assert (tmp_path / f"booster_{horizon}m.ubj").is_file()
        assert booster_device(loaded[index]) == "cpu"
        np.testing.assert_array_equal(loaded[index].predict(matrix), expected)

    cuda_rebound = load_boosters(
        tmp_path, requested_device="cuda", expected_sha256=hashes
    )
    assert all(
        booster_device(cuda_rebound[index]).startswith("cuda")
        for index in range(len(HORIZONS))
    )


def test_xgboost_booster_hash_validation_rejects_mutation(tmp_path: Path) -> None:
    booster, _ = _train_cpu_xgboost_test_model()
    hashes = save_boosters(tmp_path, {index: booster for index in range(len(HORIZONS))})
    path = tmp_path / "booster_30m.ubj"
    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA256 mismatch for horizon 30m"):
        validate_booster_hashes(tmp_path, hashes)


def test_xgboost_booster_hash_validation_rejects_missing_file(
    tmp_path: Path,
) -> None:
    booster, _ = _train_cpu_xgboost_test_model()
    hashes = save_boosters(tmp_path, {index: booster for index in range(len(HORIZONS))})
    (tmp_path / "booster_60m.ubj").unlink()
    with pytest.raises(ValueError, match="exactly three horizon booster files"):
        validate_booster_hashes(tmp_path, hashes)


def test_xgboost_booster_hash_validation_rejects_malformed_maps(
    tmp_path: Path,
) -> None:
    booster, _ = _train_cpu_xgboost_test_model()
    hashes = save_boosters(tmp_path, {index: booster for index in range(len(HORIZONS))})
    invalid_maps = (
        {},
        {**hashes, "999m": "0" * 64},
        {**hashes, "30m": "not-a-sha256"},
    )
    for invalid in invalid_maps:
        with pytest.raises(ValueError, match="SHA256"):
            validate_booster_hashes(tmp_path, invalid)


def _completed_xgboost_manifest(
    feature_store: Path, booster_sha256: dict[str, str]
) -> dict[str, object]:
    return {
        "status": "completed",
        "model_name": "xgboost",
        "model_family": "xgboost",
        "optimizer_variant": None,
        "architecture_constants": None,
        "parameter_count": None,
        "peer_features": peer_feature_metadata("xgboost", None, "none"),
        "compile": None,
        "bf16": None,
        "seed": 11,
        "git_commit_sha": "a" * 40,
        **_feature_manifest(feature_store),
        "global_context": "enabled",
        "resolved_feature_store_path": str(feature_store),
        "xgboost": {
            "version": XGBOOST_VERSION,
            "device": XGBOOST_DEVICE,
            "objective": "reg:pseudohubererror",
            "fixed_parameters": dict(XGBOOST_FIXED_PARAMETERS),
            "selected_settings": {
                "feature_store": str(feature_store),
                "fixed_parameters": dict(XGBOOST_FIXED_PARAMETERS),
            },
            "native_cuda_qualification": {
                "passed": True,
                "device": "cuda:0",
                "exact_reload_prediction_equality": True,
            },
            "booster_sha256": booster_sha256,
        },
    }


def test_completed_xgboost_manifest_requires_bound_booster_hashes(
    tmp_path: Path,
) -> None:
    feature_store = tmp_path / "store"
    feature_store.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    booster, _ = _train_cpu_xgboost_test_model()
    hashes = save_boosters(run_dir, {index: booster for index in range(len(HORIZONS))})
    manifest = _completed_xgboost_manifest(feature_store, hashes)
    validated_hashes, explicit_identity = _validate_xgboost_identity(
        manifest, feature_store, run_dir
    )
    assert validated_hashes == hashes
    assert explicit_identity.source == "explicit_registry_metadata"

    legacy_manifest = copy.deepcopy(manifest)
    del legacy_manifest["context_ablation"]
    legacy_hashes, legacy_identity = _validate_xgboost_identity(
        legacy_manifest, feature_store, run_dir
    )
    assert legacy_hashes == hashes
    assert legacy_identity.metadata == get_context_ablation("none").metadata()
    assert legacy_identity.source == "legacy_implicit_none"

    malformed_ablation = copy.deepcopy(manifest)
    malformed_ablation["context_ablation"] = None
    with pytest.raises(ValueError, match="explicit object"):
        _validate_xgboost_identity(malformed_ablation, feature_store, run_dir)

    with pytest.raises(ValueError, match="Ablation-named"):
        _validate_xgboost_identity(
            legacy_manifest,
            feature_store,
            tmp_path / "xgboost_ablation-drop_win_seed29_legacy",
        )

    missing_hashes = copy.deepcopy(manifest)
    del missing_hashes["xgboost"]["booster_sha256"]  # type: ignore[index]
    with pytest.raises(ValueError, match="SHA256 metadata is missing"):
        _validate_xgboost_identity(missing_hashes, feature_store, run_dir)


def test_native_cuda_xgboost_qualification_and_cache_cleanup(
    tmp_path: Path,
) -> None:
    if (
        sys.platform != "linux"
        or not torch.cuda.is_available()
        or not bool(xgb.build_info().get("USE_CUDA"))
    ):
        pytest.skip("native CUDA async-pool XGBoost runtime is unavailable")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    report = qualify_native_cuda_xgboost(work_dir)
    assert report["passed"] is True
    assert report["exact_reload_prediction_equality"] is True
    assert str(report["device"]).startswith("cuda")
    assert not any(work_dir.iterdir())
