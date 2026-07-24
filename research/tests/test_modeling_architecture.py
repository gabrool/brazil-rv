from __future__ import annotations

import pytest
import torch
from torch import nn

import brazil_rv.modeling.engine as engine_module

from brazil_rv.modeling.contract import (
    ABSOLUTE_PATCH_COUNT,
    COMPILE_STEADY_STATE_PASS_COUNT,
    COMPILE_WARMUP_PASS_COUNT,
    EQUITY_COUNT,
    INSTRUMENT_COUNT,
    PATCH_INPUT_WIDTH,
    RUNTIME_PROFILES,
)
from brazil_rv.modeling.engine import compile_model, warmup_compiled_model
from brazil_rv.modeling.layers import MultiHeadAttention, RotaryEmbedding
from brazil_rv.modeling.model import (
    CrossAssetPatchITransformerV1,
    count_trainable_parameters,
)


def _inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    patches = 0.1 * torch.randn(
        1,
        INSTRUMENT_COUNT,
        ABSOLUTE_PATCH_COUNT,
        PATCH_INPUT_WIDTH,
        generator=generator,
    )
    history = torch.zeros(1, INSTRUMENT_COUNT, ABSOLUTE_PATCH_COUNT, dtype=torch.bool)
    instrument = torch.zeros(1, INSTRUMENT_COUNT, dtype=torch.bool)
    instrument[:, :4] = True
    instrument[:, EQUITY_COUNT:] = True
    history[:, :4, 12:15] = True
    history[:, EQUITY_COUNT:, :15] = True
    slow = 0.1 * torch.randn(1, INSTRUMENT_COUNT, 3, generator=generator)
    return {
        "patches": patches,
        "history_patch_mask": history,
        "instrument_mask": instrument,
        "slow_features": slow,
        "state_position": torch.tensor([15]),
    }


def _forward(
    model: CrossAssetPatchITransformerV1,
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    return model(
        inputs["patches"],
        inputs["history_patch_mask"],
        inputs["instrument_mask"],
        inputs["slow_features"],
        inputs["state_position"],
    )


@pytest.fixture(scope="module")
def full_model() -> CrossAssetPatchITransformerV1:
    return CrossAssetPatchITransformerV1("full").eval()


@pytest.fixture(scope="module")
def temporal_model() -> CrossAssetPatchITransformerV1:
    return CrossAssetPatchITransformerV1("temporal_only").eval()


def test_forward_shape_and_finiteness(
    full_model: CrossAssetPatchITransformerV1,
) -> None:
    with torch.no_grad():
        output = _forward(full_model, _inputs())
    assert output.shape == (1, EQUITY_COUNT, 3)
    assert torch.isfinite(output).all()


def test_full_parameter_count(
    full_model: CrossAssetPatchITransformerV1,
) -> None:
    assert 6_300_000 <= count_trainable_parameters(full_model) <= 6_600_000


def test_equity_permutation_equivariance(
    full_model: CrossAssetPatchITransformerV1,
) -> None:
    inputs = _inputs()
    permutation = torch.arange(EQUITY_COUNT - 1, -1, -1)
    permuted = {key: value.clone() for key, value in inputs.items()}
    for key in (
        "patches",
        "history_patch_mask",
        "instrument_mask",
        "slow_features",
    ):
        permuted[key][:, :EQUITY_COUNT] = inputs[key][:, :EQUITY_COUNT][:, permutation]
    with torch.no_grad():
        baseline = _forward(full_model, inputs)
        changed = _forward(full_model, permuted)
    torch.testing.assert_close(changed, baseline[:, permutation], atol=2e-5, rtol=2e-5)


def test_inactive_equity_isolation(
    full_model: CrossAssetPatchITransformerV1,
) -> None:
    inputs = _inputs()
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["patches"][:, 10] = 1_000.0
    changed["slow_features"][:, 10] = 1_000.0
    with torch.no_grad():
        baseline_output = _forward(full_model, inputs)
        changed_output = _forward(full_model, changed)
    torch.testing.assert_close(
        baseline_output[:, :4], changed_output[:, :4], atol=0.0, rtol=0.0
    )


def test_temporal_only_isolation(
    temporal_model: CrossAssetPatchITransformerV1,
) -> None:
    inputs = _inputs()
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["patches"][:, 1] += 50.0
    changed["slow_features"][:, 1] += 50.0
    changed["patches"][:, EQUITY_COUNT:] -= 50.0
    changed["slow_features"][:, EQUITY_COUNT:] -= 50.0
    with torch.no_grad():
        baseline = _forward(temporal_model, inputs)
        mutated = _forward(temporal_model, changed)
    torch.testing.assert_close(baseline[:, 0], mutated[:, 0], atol=0.0, rtol=0.0)


def test_rope_norm_and_dynamic_state_positions() -> None:
    rope = RotaryEmbedding(head_dim=32, max_positions=70, base=10_000.0)
    generator = torch.Generator().manual_seed(3)
    query = torch.randn(2, 8, 70, 32, generator=generator)
    key = torch.randn(2, 8, 70, 32, generator=generator)
    query[1] = query[0]
    key[1] = key[0]
    positions = torch.arange(70).repeat(2, 1)
    positions[0, -1] = 15
    positions[1, -1] = 69
    rotated_query, rotated_key = rope(query, key, positions)
    torch.testing.assert_close(
        torch.linalg.vector_norm(rotated_query, dim=-1),
        torch.linalg.vector_norm(query, dim=-1),
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(rotated_key, dim=-1),
        torch.linalg.vector_norm(key, dim=-1),
    )
    assert not torch.equal(rotated_query[0, :, -1], rotated_query[1, :, -1])


def test_sdpa_boolean_key_mask_semantics() -> None:
    torch.manual_seed(5)
    attention = MultiHeadAttention(
        d_model=8, heads=2, qk_norm_eps=1e-6, rope=None
    ).eval()
    inputs = torch.randn(1, 3, 8)
    key_mask = torch.tensor([[True, False, True]])
    changed_masked = inputs.clone()
    changed_masked[:, 1] += 1_000.0
    with torch.no_grad():
        baseline = attention(inputs, key_mask)
        masked_change = attention(changed_masked, key_mask)
    torch.testing.assert_close(
        baseline[:, (0, 2)], masked_change[:, (0, 2)], atol=0.0, rtol=0.0
    )

    only_first_key = torch.tensor([[True, False, False]])
    changed_allowed = inputs.clone()
    changed_allowed[:, 0] += 10.0
    with torch.no_grad():
        allowed_baseline = attention(inputs, only_first_key)
        allowed_change = attention(changed_allowed, only_first_key)
    assert not torch.equal(allowed_baseline[:, 2], allowed_change[:, 2])


def test_inactive_predictions_are_exactly_zero(
    full_model: CrossAssetPatchITransformerV1,
) -> None:
    inputs = _inputs()
    with torch.no_grad():
        output = _forward(full_model, inputs)
    inactive = ~inputs["instrument_mask"][:, :EQUITY_COUNT]
    assert torch.equal(output[inactive], torch.zeros_like(output[inactive]))


def test_compile_warmup_reports_final_three_pass_medians(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_times = iter((11.0, 7.0, 5.0, 9.0, 6.0))
    evaluation_times = iter((13.0, 8.0, 4.0, 2.0, 3.0))
    training_calls = 0
    evaluation_calls = 0
    training_modes: list[bool] = []
    evaluation_modes: list[bool] = []

    def timed_training(model: nn.Module, *_: object) -> float:
        nonlocal training_calls
        training_calls += 1
        training_modes.append(model.training)
        return next(training_times)

    def timed_evaluation(model: nn.Module, *_: object) -> float:
        nonlocal evaluation_calls
        evaluation_calls += 1
        evaluation_modes.append(model.training)
        return next(evaluation_times)

    monkeypatch.setattr(engine_module, "_to_cuda", lambda batch: batch)
    monkeypatch.setattr(engine_module, "_timed_training_warmup_pass", timed_training)
    monkeypatch.setattr(
        engine_module, "_timed_evaluation_warmup_pass", timed_evaluation
    )
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 101)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 202)

    model = nn.Linear(1, 1)
    report = warmup_compiled_model(model, {}, {})

    assert COMPILE_WARMUP_PASS_COUNT == 5
    assert COMPILE_STEADY_STATE_PASS_COUNT == 3
    assert training_calls == 5
    assert evaluation_calls == 5
    assert training_modes == [True] * 5
    assert evaluation_modes == [False] * 5
    assert not model.training
    assert report.training_pass_seconds == (11.0, 7.0, 5.0, 9.0, 6.0)
    assert report.training_steady_state_median_seconds == 6.0
    assert report.evaluation_pass_seconds == (13.0, 8.0, 4.0, 2.0, 3.0)
    assert report.evaluation_steady_state_median_seconds == 3.0
    assert report.peak_allocated_cuda_memory_bytes == 101
    assert report.peak_reserved_cuda_memory_bytes == 202


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_in_place_compile_preserves_state_dict_keys() -> None:
    model = CrossAssetPatchITransformerV1("temporal_only").to("cuda")
    expected_keys = set(model.state_dict())
    compile_model(model, RUNTIME_PROFILES["a10"])
    assert set(model.state_dict()) == expected_keys
