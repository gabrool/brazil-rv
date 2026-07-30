from __future__ import annotations

import io

import pytest
import torch
from torch import nn

import brazil_rv.modeling.engine as engine_module
from brazil_rv.modeling.baselines import ResidualTabularMLP, SharedCausalTCN
from brazil_rv.modeling.contract import (
    ABSOLUTE_PATCH_COUNT,
    BASELINE_TCN_SETTINGS,
    COMPILE_STEADY_STATE_PASS_COUNT,
    COMPILE_WARMUP_PASS_COUNT,
    EQUITY_COUNT,
    GH200_RUNTIME,
    INSTRUMENT_COUNT,
    NEURAL_MODELS,
    PATCH_INPUT_WIDTH,
    POOLED_INDUCING_TOKEN_COUNT,
    SLOW_FEATURE_COUNT,
    SUPPORTED_MODELS,
    TABULAR_FEATURE_COUNT,
    TCN_KERNEL_SIZE,
    architecture_for_model,
    expected_trainable_parameter_count,
)
from brazil_rv.modeling.engine import compile_model, warmup_compiled_model
from brazil_rv.modeling.layers import (
    CrossAttention,
    MultiHeadAttention,
    PooledMarketMemory,
    RotaryEmbedding,
    TargetedFusionBlock,
)
from brazil_rv.modeling.model import (
    TargetedCrossAssetTransformer,
    build_neural_model,
    count_trainable_parameters,
)


BASELINE_TCN_ARCHITECTURE = architecture_for_model("tcn", BASELINE_TCN_SETTINGS)


def _build_model(model_name: str) -> nn.Module:
    return build_neural_model(
        model_name,
        BASELINE_TCN_ARCHITECTURE if model_name == "tcn" else None,
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
    slow = 0.1 * torch.randn(
        1, INSTRUMENT_COUNT, SLOW_FEATURE_COUNT, generator=generator
    )
    tabular = 0.1 * torch.randn(
        1, EQUITY_COUNT, TABULAR_FEATURE_COUNT, generator=generator
    )
    return {
        "patches": patches,
        "history_patch_mask": history,
        "instrument_mask": instrument,
        "slow_features": slow,
        "state_position": torch.tensor([15]),
        "tabular_features": tabular,
        "equity_mask": instrument[:, :EQUITY_COUNT].clone(),
    }


def _forward(
    model: nn.Module, model_name: str, inputs: dict[str, torch.Tensor]
) -> torch.Tensor:
    if model_name == "mlp":
        return model(inputs["tabular_features"], inputs["equity_mask"])
    return model(
        inputs["patches"],
        inputs["history_patch_mask"],
        inputs["instrument_mask"],
        inputs["slow_features"],
        inputs["state_position"],
    )


@pytest.fixture(scope="module", params=NEURAL_MODELS)
def neural_model(request: pytest.FixtureRequest) -> tuple[str, nn.Module]:
    model_name = str(request.param)
    return model_name, _build_model(model_name).eval()


def test_exact_public_model_contract() -> None:
    assert SUPPORTED_MODELS == (
        "temporal_only",
        "context_only",
        "pooled_market",
        "context_pooled",
        "tcn",
        "mlp",
        "xgboost",
    )
    assert NEURAL_MODELS == SUPPORTED_MODELS[:-1]
    assert PATCH_INPUT_WIDTH == 130
    assert SLOW_FEATURE_COUNT == 32
    assert TABULAR_FEATURE_COUNT == 871


def test_forward_shape_finiteness_and_parameter_count(
    neural_model: tuple[str, nn.Module],
) -> None:
    model_name, model = neural_model
    with torch.no_grad():
        output = _forward(model, model_name, _inputs())
    assert output.shape == (1, EQUITY_COUNT, 3)
    assert torch.isfinite(output).all()
    architecture = architecture_for_model(
        model_name,
        BASELINE_TCN_SETTINGS if model_name == "tcn" else None,
    )
    assert count_trainable_parameters(model) == expected_trainable_parameter_count(
        model_name, architecture
    )


def test_transformer_architectures_and_memory_are_exact() -> None:
    for model_name, context_tokens, pooled_tokens, fusion_blocks in (
        ("temporal_only", 0, 0, 0),
        ("context_only", 6, 0, 1),
        ("pooled_market", 0, 6, 1),
        ("context_pooled", 6, 6, 1),
    ):
        architecture = architecture_for_model(model_name)
        assert (
            architecture.d_model,
            architecture.attention_heads,
            architecture.head_dim,
            architecture.temporal_depth,
            architecture.swiglu_width,
        ) == (256, 8, 32, 2, 704)
        assert architecture.context_memory_tokens == context_tokens
        assert architecture.pooled_memory_tokens == pooled_tokens
        assert architecture.fusion_blocks == fusion_blocks
        model = TargetedCrossAssetTransformer(model_name)
        assert len(model.temporal_encoder.blocks) == 2
        assert (
            sum(isinstance(module, TargetedFusionBlock) for module in model.modules())
            == fusion_blocks
        )
        assert not any(
            "cross_asset_encoder" in name for name, _ in model.named_modules()
        )
        if pooled_tokens:
            assert isinstance(model.pooled_memory, PooledMarketMemory)
            assert model.pooled_memory.inducing_tokens.shape[0] == 4
            assert POOLED_INDUCING_TOKEN_COUNT == 4


def test_context_only_is_targeted_and_context_sensitive() -> None:
    model = TargetedCrossAssetTransformer("context_only").eval()
    inputs = _inputs()
    unrelated = {key: value.clone() for key, value in inputs.items()}
    unrelated["patches"][:, 1] += 100.0
    unrelated["slow_features"][:, 1] += 100.0
    context_changed = {key: value.clone() for key, value in inputs.items()}
    context_changed["patches"][:, EQUITY_COUNT] += 10.0
    context_changed["slow_features"][:, EQUITY_COUNT] += 10.0
    with torch.no_grad():
        baseline = _forward(model, "context_only", inputs)
        unrelated_output = _forward(model, "context_only", unrelated)
        context_output = _forward(model, "context_only", context_changed)
    torch.testing.assert_close(
        baseline[:, 0], unrelated_output[:, 0], atol=0.0, rtol=0.0
    )
    assert not torch.equal(baseline[:, 0], context_output[:, 0])


def test_pooled_market_active_and_inactive_isolation() -> None:
    model = TargetedCrossAssetTransformer("pooled_market").eval()
    inputs = _inputs()
    active_changed = {key: value.clone() for key, value in inputs.items()}
    active_changed["patches"][:, 1] += 20.0
    inactive_changed = {key: value.clone() for key, value in inputs.items()}
    inactive_changed["patches"][:, 10] += 20_000.0
    inactive_changed["slow_features"][:, 10] += 20_000.0
    with torch.no_grad():
        baseline = _forward(model, "pooled_market", inputs)
        active_output = _forward(model, "pooled_market", active_changed)
        inactive_output = _forward(model, "pooled_market", inactive_changed)
    assert not torch.equal(baseline[:, 0], active_output[:, 0])
    torch.testing.assert_close(baseline[:, :4], inactive_output[:, :4], atol=0, rtol=0)


@pytest.mark.parametrize("model_name", NEURAL_MODELS)
def test_equity_permutation_equivariance_and_inactive_zero(model_name: str) -> None:
    model = _build_model(model_name).eval()
    inputs = _inputs()
    permutation = torch.arange(EQUITY_COUNT - 1, -1, -1)
    permuted = {key: value.clone() for key, value in inputs.items()}
    if model_name == "mlp":
        permuted["tabular_features"] = inputs["tabular_features"][:, permutation]
        permuted["equity_mask"] = inputs["equity_mask"][:, permutation]
    else:
        for key in (
            "patches",
            "history_patch_mask",
            "instrument_mask",
            "slow_features",
        ):
            permuted[key][:, :EQUITY_COUNT] = inputs[key][:, :EQUITY_COUNT][
                :, permutation
            ]
    with torch.no_grad():
        baseline = _forward(model, model_name, inputs)
        changed = _forward(model, model_name, permuted)
    torch.testing.assert_close(changed, baseline[:, permutation], atol=3e-5, rtol=3e-5)
    inactive = (
        ~inputs["equity_mask"]
        if model_name == "mlp"
        else ~inputs["instrument_mask"][:, :EQUITY_COUNT]
    )
    assert torch.equal(baseline[inactive], torch.zeros_like(baseline[inactive]))


def test_tcn_is_causal_and_contains_no_attention() -> None:
    model = SharedCausalTCN(BASELINE_TCN_ARCHITECTURE).eval()
    assert BASELINE_TCN_ARCHITECTURE.kernel_size == TCN_KERNEL_SIZE
    assert (
        BASELINE_TCN_ARCHITECTURE.maximum_effective_context_receptive_field_patches
        == 69
    )
    assert not any(
        isinstance(module, (MultiHeadAttention, CrossAttention))
        for module in model.modules()
    )
    inputs = _inputs()
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["patches"][:, :, 20:] += 1_000.0
    changed["history_patch_mask"][:, :, 20:] = True
    with torch.no_grad():
        baseline = _forward(model, "tcn", inputs)
        output = _forward(model, "tcn", changed)
    torch.testing.assert_close(baseline, output, atol=0, rtol=0)


def test_mlp_has_only_feedforward_modules() -> None:
    model = ResidualTabularMLP()
    assert model.input_projection.in_features == TABULAR_FEATURE_COUNT
    assert not any(
        isinstance(module, (MultiHeadAttention, CrossAttention, nn.Conv1d))
        for module in model.modules()
    )


def test_state_dict_round_trip_for_every_neural_setting(
    neural_model: tuple[str, nn.Module],
) -> None:
    model_name, model = neural_model
    inputs = _inputs()
    with torch.no_grad():
        expected = _forward(model, model_name, inputs)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = _build_model(model_name).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        actual = _forward(restored, model_name, inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


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
    report = warmup_compiled_model(model, {}, {}, 0.1)
    assert COMPILE_WARMUP_PASS_COUNT == 5
    assert COMPILE_STEADY_STATE_PASS_COUNT == 3
    assert training_calls == evaluation_calls == 5
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
    model = TargetedCrossAssetTransformer("context_pooled").to("cuda")
    expected_keys = set(model.state_dict())
    compile_model(model, GH200_RUNTIME)
    assert set(model.state_dict()) == expected_keys
