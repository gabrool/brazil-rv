from dataclasses import replace

import numpy as np
import pytest
import torch

from brazil_rv.v2.config import ModelConfig
from brazil_rv.v2.data import V2DailyDataset, collate_v2_daily
from brazil_rv.v2.model import DailyMultiHorizonModel, count_non_fast_parameters
from brazil_rv.v2.train import (
    _model_forward,
    stage_p_model_config,
    _validate_stage_batch,
)
from test_v2_model import _inputs
from test_v2_store import _base_store


def test_slow_only_graph_removes_current_and_fast_parameters_and_inputs():
    inputs = _inputs()
    config = ModelConfig(
        slow_feature_count=32,
        current_feature_count=0,
        disable_fast_stream=True,
        dropout=0,
        compile_forward=False,
    )
    model = DailyMultiHorizonModel(config).eval()
    assert not any(
        name.startswith(("current_", "fast_", "absent_state"))
        for name in model.state_dict()
    )
    assert count_non_fast_parameters(model) == sum(
        p.numel() for p in model.parameters()
    )
    with torch.no_grad():
        output = model(*inputs, slow_feature_age_sessions=torch.zeros_like(inputs[0]))
        changed = model(
            *inputs,
            slow_feature_age_sessions=torch.zeros_like(inputs[0]),
            current_features=torch.full((2, 4, 20), float("nan")),
            fast_present=torch.ones((2, 4)),
        )
    assert torch.equal(output, changed)
    assert output.shape == (2, 4, 6)
    assert torch.isfinite(output).all()


def test_slow_only_dataset_does_not_read_intraday_or_fast_payload(
    tmp_path, monkeypatch
):
    path = _base_store(tmp_path)
    dataset = V2DailyDataset(
        path,
        [20, 21],
        stage="finetune",
        lookback=20,
        include_intraday=False,
        include_fast=False,
    )
    original = dataset.store.read

    def read(name, *args, **kwargs):
        assert not name.startswith(("intraday_", "fast_"))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(dataset.store, "read", read)
    batch = collate_v2_daily([dataset[0], dataset[1]])
    assert not any(name.startswith(("current_", "fast_", "v1_")) for name in batch)
    batch["targets"] = torch.zeros(2, 3, 5)
    batch["target_mask"] = torch.zeros(2, 3, 5, dtype=torch.bool)
    _validate_stage_batch("F", batch, slow_only=True)
    model = DailyMultiHorizonModel(
        ModelConfig(
            slow_feature_count=2,
            current_feature_count=0,
            disable_fast_stream=True,
            slow_lookback=20,
            compile_forward=False,
        )
    )
    assert _model_forward(model, batch).shape == (2, 3, 6)
    dataset.store.close()


def test_invalid_lending_sidecars_preserve_shared_parent_batch_fields_bitwise(tmp_path):
    from test_v2_training import _tracked_pretrain_loaders

    factory, selection = _tracked_pretrain_loaders(tmp_path)
    parent = factory().dataset
    parent.include_fast = False
    lending = V2DailyDataset(
        parent.store.root,
        parent.date_indices,
        stage="pretrain",
        lookback=20,
        include_fast=False,
        enabled_sidecars=("lending",),
    )
    left = collate_v2_daily([parent[0], parent[1]])
    right = collate_v2_daily([lending[0], lending[1]])
    for key, value in left.items():
        actual = right[key]
        if isinstance(value, torch.Tensor):
            assert torch.equal(actual, value), key
        else:
            assert actual == value, key
    assert not right["sidecar_lending_valid"].any()
    assert "targets" in left and "target_mask" in left
    assert not right["sidecar_lending_values"].any()
    parent.store.close()
    lending.store.close()
    selection.dataset.store.close()


def test_common_state_reads_only_decision_row_and_reaches_fusion(tmp_path, monkeypatch):
    values = np.arange(75, dtype=np.float32).reshape(25, 3)
    valid = np.ones_like(values, dtype=bool)
    valid[20, 0] = False
    path = _base_store(
        tmp_path,
        extra_arrays={
            "common_state_diagnostic_values": values,
            "common_state_diagnostic_valid": valid,
        },
    )
    dataset = V2DailyDataset(
        path,
        [20],
        stage="finetune",
        lookback=20,
        include_fast=False,
        include_common_state=True,
    )
    original = dataset.store.read

    def read(name, index, *args, **kwargs):
        if name.startswith("common_state_"):
            assert index == 20
        return original(name, index, *args, **kwargs)

    monkeypatch.setattr(dataset.store, "read", read)
    batch = collate_v2_daily([dataset[0]])
    torch.testing.assert_close(
        batch["common_state_features"],
        torch.tensor([[0.0, 61.0, 62.0]]),
        rtol=0,
        atol=0,
    )
    model = DailyMultiHorizonModel(
        ModelConfig(
            slow_feature_count=2,
            slow_lookback=20,
            common_state_feature_count=3,
            disable_fast_stream=True,
            dropout=0,
            compile_forward=False,
        )
    )
    captured = []
    handle = model.fusion_projection.register_forward_pre_hook(
        lambda _m, args: captured.append(args[0])
    )
    _model_forward(model, batch).sum().backward()
    handle.remove()
    assert torch.equal(
        captured[0][..., -3:], batch["common_state_features"][:, None].expand(-1, 3, -1)
    )
    assert model.fusion_projection.weight.grad[:, -2:].abs().sum() > 0
    dataset.store.close()


def test_h_and_p_reuse_uniform_stage_p_contract_but_graph_arms_do_not():
    parent = ModelConfig(slow_feature_count=32, disable_fast_stream=True)
    expected = stage_p_model_config(parent)
    assert (
        stage_p_model_config(
            replace(
                parent,
                horizon_loss_weights=tuple(x / 3.5 for x in (0.25, 0.25, 1, 1, 1)),
            )
        )
        == expected
    )
    assert stage_p_model_config(replace(parent, lambda_persistence=0.1)) == expected
    assert (
        stage_p_model_config(replace(parent, selection_horizons=(1, 2, 3, 5)))
        == expected
    )
    for changed in (
        replace(parent, current_feature_count=0),
        replace(parent, slow_feature_count=36),
        replace(parent, common_state_feature_count=3),
    ):
        assert stage_p_model_config(changed) != expected
    assert (
        ModelConfig(
            slow_feature_count=32, horizon_loss_weights=[0.2] * 5
        ).horizon_loss_weights
        == (0.2,) * 5
    )
    with pytest.raises(ValueError, match="sum to one"):
        replace(parent, horizon_loss_weights=(1, 1, 1, 1, 1))


@pytest.mark.parametrize("graph", ["S0", "C"])
def test_new_stage_p_graph_trains_and_transfers_to_same_graph(tmp_path, graph):
    from brazil_rv.v2.train import train_stage, load_pretrain_handoff
    from brazil_rv.v2.artifacts import sha256_file
    from test_v2_training import _tracked_pretrain_loaders

    factory, selection = _tracked_pretrain_loaders(tmp_path)
    training = factory()
    for dataset in (training.dataset, selection.dataset):
        dataset.include_intraday = graph != "S0"
        dataset.include_fast = False
        dataset.include_common_state = graph == "C"
    config = ModelConfig(
        slow_feature_count=32,
        current_feature_count=0 if graph == "S0" else 20,
        common_state_feature_count=3 if graph == "C" else 0,
        slow_lookback=20,
        disable_fast_stream=True,
        compile_forward=False,
    )
    result = train_stage(
        stage="P",
        seed=61,
        fold="pretrain_internal",
        train_loader=training,
        selection_loader=selection,
        output_dir=tmp_path / "p",
        model_config=stage_p_model_config(config),
        maximum_epochs=1,
        patience=1,
        device=torch.device("cpu"),
    )
    model = DailyMultiHorizonModel(config)
    transferred = load_pretrain_handoff(
        model,
        result.raw_patience_checkpoint,
        expected_sha256=sha256_file(result.raw_patience_checkpoint),
        expected_seed=61,
    )
    assert transferred == {
        name
        for name in dict(model.named_parameters())
        if not name.startswith("fast_encoder.")
    }
    for dataset in (training.dataset, selection.dataset):
        dataset.store.close()
