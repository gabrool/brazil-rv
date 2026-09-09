from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from brazil_rv.modeling.model import SharedCausalTCN
from brazil_rv.v2.config import ModelConfig
from brazil_rv.v2.data import collate_v2_daily
from brazil_rv.v2.model import (
    DailyMultiHorizonModel,
    count_non_fast_parameters,
    load_v1_fast_encoder,
)


def test_fast_off_bypasses_tcn_and_matches_explicit_absent_stream() -> None:
    inputs = _inputs()
    config = ModelConfig(slow_feature_count=32, dropout=0.0, compile_forward=False)
    reference = DailyMultiHorizonModel(config).eval()
    ablated = DailyMultiHorizonModel(replace(config, disable_fast_stream=True)).eval()
    ablated.load_state_dict(reference.state_dict())
    calls = []
    handle = ablated.fast_encoder.register_forward_hook(lambda *_: calls.append(1))
    try:
        with torch.no_grad():
            expected = _run(reference, *inputs, fast_present=torch.zeros(2, 4))
            actual = _run(
                ablated,
                *inputs,
                fast_present=torch.ones(2, 4),
                fast_patch_values=torch.full((2, 4, 5, 7), float("nan")),
                fast_name_index=torch.arange(4).expand(2, -1),
            )
        assert torch.equal(actual, expected)
        assert not calls
    finally:
        handle.remove()


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    slow = torch.randn(2, 4, 60, 32)
    feature_mask = torch.ones_like(slow, dtype=torch.bool)
    history = torch.ones(2, 4, 60, dtype=torch.bool)
    active = torch.tensor([[True, True, True, False], [True, True, False, False]])
    return slow, feature_mask, history, active


def _run(
    model: DailyMultiHorizonModel,
    slow: torch.Tensor,
    feature_mask: torch.Tensor,
    history: torch.Tensor,
    active: torch.Tensor,
    *args: object,
    **kwargs: object,
) -> torch.Tensor:
    slow_age = kwargs.pop(
        "slow_feature_age_sessions",
        torch.where(feature_mask, 0.0, -1.0),
    )
    current = torch.zeros(
        (*slow.shape[:2], model.config.current_feature_count), dtype=slow.dtype
    )
    current_mask = torch.ones_like(current, dtype=torch.bool)
    current_age = kwargs.pop(
        "current_feature_age_sessions", torch.zeros_like(current)
    )
    return model(
        slow,
        feature_mask,
        history,
        active,
        *args,
        current_features=current,
        current_feature_mask=current_mask,
        slow_feature_age_sessions=slow_age,
        current_feature_age_sessions=current_age,
        **kwargs,
    )


def _compact_sample(fast_count: int, *, offset: int = 0) -> dict[str, object]:
    values = np.arange(fast_count * 5 * 7, dtype=np.float32).reshape(
        fast_count, 5, 7
    )
    valid = np.ones_like(values, dtype=np.bool_)
    return {
        "date_index": np.int64(offset),
        "fast_patch_values": values,
        "fast_patch_valid": valid,
        "fast_patch_mask": np.ones((fast_count, 5), dtype=np.bool_),
        "fast_name_index": np.arange(offset, offset + fast_count, dtype=np.int64),
        "fast_state_position": np.full(fast_count, 5, dtype=np.int64),
        "v1_equity_slow": np.zeros((fast_count, 32), dtype=np.float32),
    }


def test_compact_fast_collate_pads_only_present_name_slots() -> None:
    batch = collate_v2_daily(
        (_compact_sample(2), _compact_sample(1, offset=2))
    )
    assert batch["fast_patch_values"].shape == (2, 2, 5, 7)
    assert batch["fast_patch_valid"].shape == (2, 2, 5, 7)
    assert batch["fast_name_index"].tolist() == [[0, 1], [2, -1]]
    assert not batch["fast_patch_mask"][1, 1].any()
    assert not batch["fast_patch_values"][1, 1].any()


def test_all_absent_collate_allocates_zero_fast_names() -> None:
    batch = collate_v2_daily((_compact_sample(0), _compact_sample(0)))
    assert batch["fast_patch_values"].shape == (2, 0, 5, 7)
    assert batch["fast_patch_valid"].numel() == 0
    assert batch["fast_name_index"].shape == (2, 0)


@pytest.mark.parametrize("layers", [1, 2])
def test_model_shape_zero_to_close_and_parameter_cap(layers: int) -> None:
    model = DailyMultiHorizonModel(
        ModelConfig(slow_feature_count=32, gru_layers=layers)
    )
    slow, feature_mask, history, active = _inputs()
    predictions = _run(model, slow, feature_mask, history, active)
    assert predictions.shape == (2, 4, 6)
    assert torch.count_nonzero(predictions[..., 5]) == 0
    assert torch.count_nonzero(predictions[~active]) == 0
    assert count_non_fast_parameters(model) <= 200_000
    assert not any(isinstance(module, nn.Embedding) for module in model.modules())
    assert torch.count_nonzero(model.fast_gate.weight) == 0
    assert torch.count_nonzero(model.pool_gate.weight) == 0


def test_absent_fast_path_ignores_patches_and_receives_gradient() -> None:
    model = DailyMultiHorizonModel(ModelConfig(slow_feature_count=32)).eval()
    slow, feature_mask, history, active = _inputs()
    absent = torch.zeros(2, 4)
    first = _run(
        model,
        slow,
        feature_mask,
        history,
        active,
        fast_patch_values=torch.empty(2, 0, 69, 7),
        fast_patch_valid=torch.empty(2, 0, 69, 7, dtype=torch.bool),
        fast_patch_mask=torch.empty(2, 0, 69, dtype=torch.bool),
        fast_name_index=torch.empty(2, 0, dtype=torch.long),
        fast_state_position=torch.empty(2, 0, dtype=torch.long),
        fast_present=absent,
    )
    second = _run(model, slow, feature_mask, history, active, fast_present=absent)
    assert torch.equal(first, second)
    first[..., :5].sum().backward()
    assert model.absent_state.grad is not None
    assert torch.isfinite(model.absent_state.grad).all()


def test_active_name_with_empty_slow_history_uses_zero_initial_state() -> None:
    model = DailyMultiHorizonModel(ModelConfig(slow_feature_count=32)).eval()
    slow, feature_mask, history, active = _inputs()
    history[0, 1] = False
    feature_mask[0, 1] = False
    changed = slow.clone()
    changed[0, 1] = 1_000.0

    slow_state = model._slow_states(
        slow,
        feature_mask,
        history,
        torch.zeros_like(slow),
    )

    first = _run(model, slow, feature_mask, history, active)
    second = _run(model, changed, feature_mask, history, active)

    assert torch.count_nonzero(slow_state[0, 1]) == 0
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert torch.count_nonzero(first[0, 1]) > 0

    tracked = slow.clone().requires_grad_()
    _run(model, tracked, feature_mask, history, active).sum().backward()
    assert tracked.grad is not None
    assert torch.isfinite(tracked.grad).all()
    assert torch.count_nonzero(tracked.grad[0, 1]) == 0


def test_slow_feature_mask_zeroes_payload_and_remains_model_visible() -> None:
    model = DailyMultiHorizonModel(ModelConfig(slow_feature_count=32)).eval()
    slow, feature_mask, history, active = _inputs()
    feature_mask[0, 0, -1, 7] = False
    first_payload = slow.clone()
    second_payload = slow.clone()
    first_payload[0, 0, -1, 7] = -1_000_000.0
    second_payload[0, 0, -1, 7] = 1_000_000.0

    masked_first = _run(model, first_payload, feature_mask, history, active)
    masked_second = _run(model, second_payload, feature_mask, history, active)
    fully_valid = _run(
        model, second_payload, torch.ones_like(feature_mask), history, active
    )

    assert torch.equal(masked_first, masked_second)
    assert not torch.equal(masked_second[0, 0], fully_valid[0, 0])


def test_feature_age_distinguishes_known_staleness_from_unknown_provenance() -> None:
    torch.manual_seed(109)
    model = DailyMultiHorizonModel(
        ModelConfig(slow_feature_count=32, dropout=0.0)
    ).eval()
    slow, feature_mask, history, active = _inputs()
    feature_mask[0, 0, -1, 7] = False
    unknown_age = torch.where(feature_mask, 0.0, -1.0)
    known_stale_age = unknown_age.clone()
    known_stale_age[0, 0, -1, 7] = 21.0

    unknown = _run(
        model,
        slow,
        feature_mask,
        history,
        active,
        slow_feature_age_sessions=unknown_age,
    )
    stale = _run(
        model,
        slow,
        feature_mask,
        history,
        active,
        slow_feature_age_sessions=known_stale_age,
    )

    assert not torch.equal(unknown[0, 0], stale[0, 0])


def test_feature_age_contract_rejects_nonfinite_misaligned_and_unknown_valid() -> None:
    model = DailyMultiHorizonModel(ModelConfig(slow_feature_count=32)).eval()
    slow, feature_mask, history, active = _inputs()
    with pytest.raises(ValueError, match="slow feature ages are misaligned"):
        _run(
            model,
            slow,
            feature_mask,
            history,
            active,
            slow_feature_age_sessions=torch.zeros(2, 4, 60, 31),
        )

    nonfinite = torch.zeros_like(slow)
    nonfinite[0, 0, -1, 0] = torch.nan
    with pytest.raises(RuntimeError, match="feature ages must be finite"):
        _run(
            model,
            slow,
            feature_mask,
            history,
            active,
            slow_feature_age_sessions=nonfinite,
        )

    unknown_valid = torch.zeros_like(slow)
    unknown_valid[0, 0, -1, 0] = -1.0
    with pytest.raises(RuntimeError, match="known for every valid feature"):
        _run(
            model,
            slow,
            feature_mask,
            history,
            active,
            slow_feature_age_sessions=unknown_valid,
        )


def test_nan_masking_covers_empty_slow_and_fast_histories() -> None:
    model = DailyMultiHorizonModel(
        ModelConfig(slow_feature_count=32, slow_lookback=60)
    ).eval()
    slow, feature_mask, history, active = _inputs()
    slow[0, 1] = torch.nan
    feature_mask[0, 1] = False
    history[0, 1] = False
    patches = torch.randn(2, 4, 69, 7)
    patch_valid = torch.ones_like(patches, dtype=torch.bool)
    patch_mask = torch.ones(2, 4, 69, dtype=torch.bool)
    patches[0, 1] = torch.nan
    patch_valid[0, 1] = False
    present = torch.ones(2, 4, dtype=torch.bool)
    name_index = torch.arange(4)[None, :].expand(2, -1)
    state_position = torch.full((2, 4), 69, dtype=torch.long)

    with torch.no_grad():
        predictions = _run(
            model,
            slow,
            feature_mask,
            history,
            active,
            fast_patch_values=patches,
            fast_patch_valid=patch_valid,
            fast_patch_mask=patch_mask,
            fast_name_index=name_index,
            fast_state_position=state_position,
            fast_present=present,
        )

    assert torch.isfinite(predictions[active]).all()


def test_fast_encoder_runs_only_collated_compact_slots() -> None:
    model = DailyMultiHorizonModel(ModelConfig(slow_feature_count=32)).eval()
    slow, feature_mask, history, active = _inputs()
    patches = torch.randn(2, 2, 69, 7)
    patch_valid = torch.ones_like(patches, dtype=torch.bool)
    patch_mask = torch.ones(2, 2, 69, dtype=torch.bool)
    patch_mask[1, 1] = False
    patch_valid[1, 1] = False
    patches[1, 1] = 0.0
    name_index = torch.tensor([[0, 2], [1, -1]])
    present = torch.tensor(
        [[True, False, True, False], [False, True, False, False]]
    )
    state_position = torch.tensor([[69, 69], [69, 0]])
    observed: list[tuple[torch.Size, torch.Tensor]] = []

    def capture_inputs(
        _module: nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        observed.append((inputs[0].shape, inputs[3].detach().clone()))

    handle = model.fast_encoder.register_forward_pre_hook(capture_inputs)
    try:
        _run(
            model,
            slow,
            feature_mask,
            history,
            active,
            fast_patch_values=patches,
            fast_patch_valid=patch_valid,
            fast_patch_mask=patch_mask,
            fast_name_index=name_index,
            fast_state_position=state_position,
            fast_present=present,
        )
    finally:
        handle.remove()

    assert len(observed) == 1
    shape, selected_position = observed[0]
    assert shape == torch.Size((4, 1, 69, 7))
    assert selected_position.tolist() == [69, 69, 69, 0]


def test_compact_legacy_fast_path_matches_dense_adapter() -> None:
    torch.manual_seed(59)
    config = ModelConfig(
        slow_feature_count=32,
        dropout=0.0,
        fast_encoder_mode="legacy_v1_contaminated",
        allow_contaminated_v1_initialization=True,
    )
    model = DailyMultiHorizonModel(config).eval()

    slow = torch.randn(2, 3, 60, 32)
    feature_mask = torch.ones_like(slow, dtype=torch.bool)
    history = torch.ones(2, 3, 60, dtype=torch.bool)
    active = torch.ones(2, 3, dtype=torch.bool)
    patches = torch.randn(2, 3, 69, 130)
    patch_mask = torch.rand(2, 3, 69) > 0.15
    present = torch.tensor([[True, False, True], [False, True, False]])
    v1_slow = torch.randn(2, 3, 32)
    dense = _run(
        model,
        slow,
        feature_mask,
        history,
        active,
        patches,
        patch_mask,
        present,
        fast_state_position=torch.full((2,), 81, dtype=torch.long),
        v1_equity_slow=v1_slow,
    )
    compact_values = torch.zeros(2, 2, 69, 130)
    compact_valid = torch.zeros_like(compact_values, dtype=torch.bool)
    compact_mask = torch.zeros(2, 2, 69, dtype=torch.bool)
    compact_slow = torch.zeros(2, 2, 32)
    compact_names = torch.tensor([[0, 2], [1, -1]])
    compact_position = torch.tensor([[81, 81], [81, 0]])
    for batch, names in enumerate(((0, 2), (1,))):
        for slot, name in enumerate(names):
            compact_values[batch, slot] = patches[batch, name]
            compact_mask[batch, slot] = patch_mask[batch, name]
            compact_valid[batch, slot] = patch_mask[batch, name, :, None]
            compact_slow[batch, slot] = v1_slow[batch, name]
    compact = _run(
        model,
        slow,
        feature_mask,
        history,
        active,
        fast_patch_values=compact_values,
        fast_patch_valid=compact_valid,
        fast_patch_mask=compact_mask,
        fast_name_index=compact_names,
        fast_state_position=compact_position,
        fast_present=present,
        v1_equity_slow=compact_slow,
    )
    torch.testing.assert_close(compact, dense, rtol=1e-5, atol=2e-6)


def test_native_fast_path_supports_variable_completed_patch_counts() -> None:
    model = DailyMultiHorizonModel(ModelConfig(slow_feature_count=32)).eval()
    slow, feature_mask, history, active = _inputs()
    patches = torch.randn(2, 2, 5, 7)
    patch_valid = torch.ones_like(patches, dtype=torch.bool)
    patch_mask = torch.tensor(
        [
            [[True, True, True, False, False], [True, True, True, True, True]],
            [[True, True, True, True, False], [False, False, False, False, False]],
        ]
    )
    patch_valid &= patch_mask[..., None]
    name_index = torch.tensor([[0, 2], [1, -1]])
    state_position = torch.tensor([[3, 5], [4, 0]])
    present = torch.tensor(
        [[True, False, True, False], [False, True, False, False]]
    )
    first = _run(
        model,
        slow,
        feature_mask,
        history,
        active,
        fast_patch_values=patches,
        fast_patch_valid=patch_valid,
        fast_patch_mask=patch_mask,
        fast_name_index=name_index,
        fast_state_position=state_position,
        fast_present=present,
    )
    changed = patches.clone()
    changed[~patch_mask] = 1_000_000.0
    second = _run(
        model,
        slow,
        feature_mask,
        history,
        active,
        fast_patch_values=changed,
        fast_patch_valid=patch_valid,
        fast_patch_mask=patch_mask,
        fast_name_index=name_index,
        fast_state_position=state_position,
        fast_present=present,
    )
    assert first.shape == (2, 4, 6)
    assert torch.equal(first, second)
    with pytest.raises(ValueError, match="frozen limit"):
        too_long_mask = torch.ones(2, 2, 70, dtype=torch.bool)
        too_long_mask[1, 1] = False
        _run(
            model,
            slow,
            feature_mask,
            history,
            active,
            fast_patch_values=torch.zeros(2, 2, 70, 7),
            fast_patch_valid=torch.zeros(2, 2, 70, 7, dtype=torch.bool),
            fast_patch_mask=too_long_mask,
            fast_name_index=name_index,
            fast_state_position=torch.tensor([[70, 70], [70, 0]]),
            fast_present=present,
        )


def test_pooling_excludes_inactive_names() -> None:
    model = DailyMultiHorizonModel(ModelConfig(slow_feature_count=32)).eval()
    slow, feature_mask, history, active = _inputs()
    reference = _run(model, slow, feature_mask, history, active)
    changed = slow.clone()
    changed[~active] = 1_000.0
    changed_history = history.clone()
    changed_history[~active] = False
    changed_feature_mask = feature_mask.clone()
    changed_feature_mask[~active] = False
    actual = _run(model, changed, changed_feature_mask, changed_history, active)
    assert torch.equal(reference[active], actual[active])


def test_left_padding_never_advances_slow_state() -> None:
    torch.manual_seed(101)
    short = DailyMultiHorizonModel(
        ModelConfig(slow_feature_count=3, slow_lookback=20, dropout=0.0)
    ).eval()
    long = DailyMultiHorizonModel(
        ModelConfig(slow_feature_count=3, slow_lookback=60, dropout=0.0)
    ).eval()
    long.load_state_dict(short.state_dict(), strict=True)
    real = torch.randn(1, 2, 5, 3)

    short_values = torch.zeros(1, 2, 20, 3)
    short_values[:, :, -5:] = real
    short_valid = torch.zeros_like(short_values, dtype=torch.bool)
    short_valid[:, :, -5:] = True
    short_history = torch.zeros(1, 2, 20, dtype=torch.bool)
    short_history[:, :, -5:] = True

    long_values = torch.zeros(1, 2, 60, 3)
    long_values[:, :, -5:] = real
    long_valid = torch.zeros_like(long_values, dtype=torch.bool)
    long_valid[:, :, -5:] = True
    long_history = torch.zeros(1, 2, 60, dtype=torch.bool)
    long_history[:, :, -5:] = True

    torch.testing.assert_close(
        short._slow_states(
            short_values,
            short_valid,
            short_history,
            torch.where(short_valid, 0.0, -1.0),
        ),
        long._slow_states(
            long_values,
            long_valid,
            long_history,
            torch.where(long_valid, 0.0, -1.0),
        ),
        rtol=0.0,
        atol=0.0,
    )


def test_real_missing_session_is_not_sequence_padding() -> None:
    torch.manual_seed(103)
    model = DailyMultiHorizonModel(
        ModelConfig(slow_feature_count=3, slow_lookback=20, dropout=0.0)
    ).eval()
    values = torch.zeros(1, 1, 20, 3)
    values[:, :, -4:] = torch.randn(1, 1, 4, 3)
    feature_valid = torch.zeros_like(values, dtype=torch.bool)
    feature_valid[:, :, -4:] = True
    padded_history = torch.zeros(1, 1, 20, dtype=torch.bool)
    padded_history[:, :, -4:] = True

    missing_history = padded_history.clone()
    missing_history[:, :, -5] = True
    feature_age = torch.where(feature_valid, 0.0, -1.0)
    missing_state = model._slow_states(
        values, feature_valid, missing_history, feature_age
    )
    padded_state = model._slow_states(
        values, feature_valid, padded_history, feature_age
    )

    assert not torch.equal(missing_state, padded_state)


def test_current_feature_mask_zeroes_payload_but_exposes_missingness() -> None:
    model = DailyMultiHorizonModel(ModelConfig(slow_feature_count=32)).eval()
    slow, feature_mask, history, active = _inputs()
    current = torch.randn(2, 4, model.config.current_feature_count)
    current_mask = torch.ones_like(current, dtype=torch.bool)
    current_mask[0, 0, 3] = False
    first = current.clone()
    second = current.clone()
    first[0, 0, 3] = -1_000_000.0
    second[0, 0, 3] = 1_000_000.0
    masked_first = model(
        slow,
        feature_mask,
        history,
        active,
        current_features=first,
        current_feature_mask=current_mask,
        slow_feature_age_sessions=torch.zeros_like(slow),
        current_feature_age_sessions=torch.where(current_mask, 0.0, -1.0),
    )
    masked_second = model(
        slow,
        feature_mask,
        history,
        active,
        current_features=second,
        current_feature_mask=current_mask,
        slow_feature_age_sessions=torch.zeros_like(slow),
        current_feature_age_sessions=torch.where(current_mask, 0.0, -1.0),
    )
    fully_valid = model(
        slow,
        feature_mask,
        history,
        active,
        current_features=second,
        current_feature_mask=torch.ones_like(current_mask),
        slow_feature_age_sessions=torch.zeros_like(slow),
        current_feature_age_sessions=torch.zeros_like(current),
    )
    assert torch.equal(masked_first, masked_second)
    assert not torch.equal(masked_second[0, 0], fully_valid[0, 0])


def test_v1_fast_checkpoint_load_is_strict_and_hash_bound(tmp_path) -> None:
    legacy = {
        "fast_encoder_mode": "legacy_v1_contaminated",
        "allow_contaminated_v1_initialization": True,
    }
    source_model = DailyMultiHorizonModel(
        ModelConfig(slow_feature_count=32, **legacy)
    )
    source = {
        name: torch.full_like(value, 0.25)
        for name, value in source_model.fast_encoder.state_dict().items()
    }
    checkpoint = tmp_path / "v1.pt"
    torch.save({"model_state_dict": source}, checkpoint)
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    target = DailyMultiHorizonModel(ModelConfig(slow_feature_count=32, **legacy))
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
            **legacy,
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
    model = DailyMultiHorizonModel(
        ModelConfig(
            slow_feature_count=32,
            fast_encoder_mode="legacy_v1_contaminated",
            allow_contaminated_v1_initialization=True,
        )
    ).eval()
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
    assert (actual - expected).abs().max().detach().item() <= 2e-6
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-6)


def test_fast_initializer_requires_external_expected_sha(tmp_path) -> None:
    source = DailyMultiHorizonModel(
        ModelConfig(
            slow_feature_count=32,
            fast_encoder_mode="legacy_v1_contaminated",
            allow_contaminated_v1_initialization=True,
        )
    )
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


def test_v1_initialization_requires_explicit_contamination_opt_in() -> None:
    with pytest.raises(ValueError, match="explicit"):
        ModelConfig(
            slow_feature_count=32,
            fast_encoder_mode="legacy_v1_contaminated",
        )
    native = DailyMultiHorizonModel(ModelConfig(slow_feature_count=32))
    assert native.fast_initialization_provenance == {
        "mode": "fresh",
        "contaminated": False,
        "explicitly_allowed": False,
        "checkpoint_sha256": None,
    }
