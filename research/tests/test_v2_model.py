from __future__ import annotations

import hashlib
from typing import cast

import pytest
import torch
from torch import nn

from brazil_rv.modeling.model import SharedCausalTCN
from brazil_rv.v2.config import ModelConfig
from brazil_rv.v2.model import (
    DailyMultiHorizonModel,
    count_non_fast_parameters,
    load_v1_fast_encoder,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    slow = torch.randn(2, 4, 60, 32)
    history = torch.ones(2, 4, 60, dtype=torch.bool)
    active = torch.tensor([[True, True, True, False], [True, True, False, False]])
    return slow, history, active


class _DenseFastReference(DailyMultiHorizonModel):
    """The pre-optimization fast branch, retained only as a test oracle."""

    def _fast_states(
        self,
        slow: torch.Tensor,
        present: torch.Tensor,
        fast_patches: torch.Tensor | None,
        fast_patch_mask: torch.Tensor | None,
        fast_state_position: torch.Tensor | None,
        v1_equity_slow: torch.Tensor | None,
    ) -> torch.Tensor:
        absent = self.absent_state.view(1, 1, -1).expand_as(slow)
        encoded = self.fast_encoder(
            cast(torch.Tensor, fast_patches),
            cast(torch.Tensor, fast_patch_mask),
            cast(torch.Tensor, v1_equity_slow),
            fast_state_position,
        )
        return torch.where(present[..., None].bool(), encoded, absent)


@pytest.mark.parametrize("layers", [1, 2])
def test_model_shape_zero_to_close_and_parameter_cap(layers: int) -> None:
    model = DailyMultiHorizonModel(
        ModelConfig(slow_feature_count=32, gru_layers=layers)
    )
    slow, history, active = _inputs()
    predictions = model(slow, history, active)
    assert predictions.shape == (2, 4, 6)
    assert torch.count_nonzero(predictions[..., 5]) == 0
    assert torch.count_nonzero(predictions[~active]) == 0
    assert count_non_fast_parameters(model) <= 150_000
    assert not any(isinstance(module, nn.Embedding) for module in model.modules())
    assert torch.count_nonzero(model.fast_gate.weight) == 0
    assert torch.count_nonzero(model.pool_gate.weight) == 0


def test_absent_fast_path_ignores_patches_and_receives_gradient() -> None:
    model = DailyMultiHorizonModel(ModelConfig(slow_feature_count=32)).eval()
    slow, history, active = _inputs()
    patches = torch.randn(2, 4, 69, 130)
    patch_mask = torch.ones(2, 4, 69, dtype=torch.bool)
    v1_slow = torch.randn(2, 4, 32)
    absent = torch.zeros(2, 4)
    first = model(
        slow, history, active, patches, patch_mask, absent, None, None, v1_slow
    )
    second = model(
        slow,
        history,
        active,
        patches + 100.0,
        patch_mask,
        absent,
        None,
        None,
        v1_slow,
    )
    assert torch.equal(first, second)
    first[..., :5].sum().backward()
    assert model.absent_state.grad is not None
    assert torch.isfinite(model.absent_state.grad).all()


def test_active_name_with_empty_slow_history_uses_zero_initial_state() -> None:
    model = DailyMultiHorizonModel(ModelConfig(slow_feature_count=32)).eval()
    slow, history, active = _inputs()
    history[0, 1] = False
    changed = slow.clone()
    changed[0, 1] = 1_000.0

    slow_state = model._slow_states(
        slow,
        history,
        torch.zeros_like(active),
        torch.ones_like(active, dtype=slow.dtype),
    )

    first = model(slow, history, active)
    second = model(changed, history, active)

    assert torch.count_nonzero(slow_state[0, 1]) == 0
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert torch.count_nonzero(first[0, 1]) > 0

    tracked = slow.clone().requires_grad_()
    model(tracked, history, active).sum().backward()
    assert tracked.grad is not None
    assert torch.isfinite(tracked.grad).all()
    assert torch.count_nonzero(tracked.grad[0, 1]) == 0


def test_fast_encoder_runs_only_present_flattened_rows() -> None:
    model = DailyMultiHorizonModel(ModelConfig(slow_feature_count=32)).eval()
    slow, history, active = _inputs()
    patches = torch.randn(2, 4, 69, 130)
    patch_mask = torch.ones(2, 4, 69, dtype=torch.bool)
    v1_slow = torch.randn(2, 4, 32)
    present = torch.tensor(
        [[True, False, True, False], [False, True, False, False]]
    )
    state_position = torch.full((2, 4), 81, dtype=torch.long)
    observed: list[tuple[torch.Size, torch.Tensor]] = []

    def capture_inputs(
        _module: nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        observed.append((inputs[0].shape, inputs[3].detach().clone()))

    handle = model.fast_encoder.register_forward_pre_hook(capture_inputs)
    try:
        model(
            slow,
            history,
            active,
            patches,
            patch_mask,
            present,
            None,
            state_position,
            v1_slow,
        )
    finally:
        handle.remove()

    assert len(observed) == 1
    shape, selected_position = observed[0]
    assert shape == torch.Size((int(present.sum()), 1, 69, 130))
    assert selected_position.shape == (int(present.sum()),)
    assert torch.all(selected_position == 81)


def test_sparse_fast_path_matches_dense_outputs_and_gradients() -> None:
    torch.manual_seed(59)
    config = ModelConfig(slow_feature_count=32, dropout=0.0)
    sparse = DailyMultiHorizonModel(config).eval()
    dense = _DenseFastReference(config).eval()
    dense.load_state_dict(sparse.state_dict())

    base_slow = torch.randn(2, 3, 60, 32)
    history = torch.ones(2, 3, 60, dtype=torch.bool)
    active = torch.ones(2, 3, dtype=torch.bool)
    base_patches = torch.randn(2, 3, 69, 130)
    patch_mask = torch.rand(2, 3, 69) > 0.15
    present = torch.tensor([[True, False, True], [False, True, False]])
    base_v1_slow = torch.randn(2, 3, 32)
    state_position = torch.full((2,), 81, dtype=torch.long)
    loss_weights = torch.randn(2, 3, 6)

    def run(
        model: DailyMultiHorizonModel,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        slow = base_slow.clone().requires_grad_()
        patches = base_patches.clone().requires_grad_()
        v1_slow = base_v1_slow.clone().requires_grad_()
        output = model(
            slow,
            history,
            active,
            patches,
            patch_mask,
            present,
            None,
            state_position,
            v1_slow,
        )
        (output * loss_weights).sum().backward()
        return output.detach(), slow.grad, patches.grad, v1_slow.grad

    sparse_result = run(sparse)
    dense_result = run(dense)
    for actual, expected in zip(sparse_result, dense_result, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)

    assert torch.count_nonzero(sparse_result[2][~present]) == 0
    assert torch.count_nonzero(sparse_result[3][~present]) == 0
    for (sparse_name, sparse_parameter), (dense_name, dense_parameter) in zip(
        sparse.named_parameters(), dense.named_parameters(), strict=True
    ):
        assert sparse_name == dense_name
        assert sparse_parameter.grad is not None
        assert dense_parameter.grad is not None
        torch.testing.assert_close(
            sparse_parameter.grad,
            dense_parameter.grad,
            rtol=2e-4,
            atol=2e-6,
            msg=lambda message: f"{sparse_name}: {message}",
        )


def test_fast_path_enforces_synthetic_cutoff_345() -> None:
    model = DailyMultiHorizonModel(ModelConfig(slow_feature_count=32)).eval()
    slow, history, active = _inputs()
    patches = torch.randn(2, 4, 69, 130)
    patch_mask = torch.ones(2, 4, 69, dtype=torch.bool)
    v1_slow = torch.randn(2, 4, 32)
    present = torch.ones(2, 4)
    assert model(
        slow, history, active, patches, patch_mask, present, None, None, v1_slow
    ).shape == (
        2,
        4,
        6,
    )
    with pytest.raises(ValueError, match="cutoff index 345"):
        model(
            slow,
            history,
            active,
            patches[:, :, :-1],
            patch_mask[:, :, :-1],
            present,
            None,
            None,
            v1_slow,
        )


def test_pooling_excludes_inactive_names() -> None:
    model = DailyMultiHorizonModel(ModelConfig(slow_feature_count=32)).eval()
    slow, history, active = _inputs()
    reference = model(slow, history, active)
    changed = slow.clone()
    changed[~active] = 1_000.0
    changed_history = history.clone()
    changed_history[~active] = False
    actual = model(changed, changed_history, active)
    assert torch.equal(reference[active], actual[active])


def test_v1_fast_checkpoint_load_is_strict_and_hash_bound(tmp_path) -> None:
    source_model = DailyMultiHorizonModel(ModelConfig(slow_feature_count=32))
    source = {
        name: torch.full_like(value, 0.25)
        for name, value in source_model.fast_encoder.state_dict().items()
    }
    checkpoint = tmp_path / "v1.pt"
    torch.save({"model_state_dict": source}, checkpoint)
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    target = DailyMultiHorizonModel(ModelConfig(slow_feature_count=32))
    initialized = load_v1_fast_encoder(target, checkpoint, expected_sha256=expected)
    assert initialized == frozenset(
        f"fast_encoder.{name}" for name, _ in target.fast_encoder.named_parameters()
    )
    assert all(
        torch.equal(value, source[name])
        for name, value in target.fast_encoder.state_dict().items()
    )
    configured = DailyMultiHorizonModel(
        ModelConfig(
            slow_feature_count=32,
            fast_pretrained=True,
            fast_pretrained_checkpoint=checkpoint,
            fast_pretrained_sha256=expected,
        )
    )
    assert configured.fast_checkpoint_sha256 == expected
    assert configured.pretrained_parameter_names == initialized
    with pytest.raises(ValueError, match="SHA-256"):
        load_v1_fast_encoder(target, checkpoint, expected_sha256="0" * 64)


def test_fast_encoder_exactly_matches_deployed_v1_instrument_state() -> None:
    torch.manual_seed(41)
    batch_size, name_count = 2, 4
    parent = SharedCausalTCN(equity_count=name_count).eval()
    model = DailyMultiHorizonModel(ModelConfig(slow_feature_count=32)).eval()
    model.fast_encoder.load_state_dict(
        {
            name: parent.state_dict()[name]
            for name in model.fast_encoder.state_dict()
        },
        strict=True,
    )
    real_patches = torch.randn(batch_size, name_count, 69, 130)
    real_mask = torch.ones(batch_size, name_count, 69, dtype=torch.bool)
    v1_slow = torch.randn(batch_size, name_count, 32)
    neutralized = v1_slow.clone()
    neutralized[
        ...,
        [1, 2, 3, 12, 13, 14, 15, 16, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29],
    ] = 0.0
    instrument_count = name_count + 15
    parent_patches = torch.zeros(batch_size, instrument_count, 81, 130)
    parent_mask = torch.zeros(batch_size, instrument_count, 81, dtype=torch.bool)
    parent_slow = torch.zeros(batch_size, instrument_count, 32)
    parent_patches[:, :name_count, 12:] = real_patches
    parent_mask[:, :name_count, 12:] = real_mask
    parent_slow[:, :name_count] = neutralized
    expected = parent._instrument_states(
        parent_patches,
        parent_mask,
        parent_slow,
        torch.full((batch_size,), 81, dtype=torch.long),
    )[:, :name_count]
    actual = model.fast_encoder(real_patches, real_mask, v1_slow)
    # CPU Conv1d selects a different batched kernel when the deployed parent
    # also carries its 15 non-equity contexts.  The independent construction
    # must nevertheless reproduce the same numerical state.
    assert (actual - expected).abs().max().detach().item() <= 1e-6
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


def test_fast_initializer_requires_external_expected_sha(tmp_path) -> None:
    source = DailyMultiHorizonModel(ModelConfig(slow_feature_count=32))
    checkpoint = tmp_path / "v1.pt"
    torch.save({"model_state_dict": source.fast_encoder.state_dict()}, checkpoint)
    with pytest.raises(ValueError, match="expected SHA-256"):
        load_v1_fast_encoder(source, checkpoint)
    with pytest.raises(ValueError, match="must be set together"):
        ModelConfig(
            slow_feature_count=32,
            fast_pretrained=True,
            fast_pretrained_checkpoint=checkpoint,
        )
