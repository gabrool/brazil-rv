from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import brazil_rv.modeling.engine as engine_module
from brazil_rv.modeling.contract import (
    ABSOLUTE_PATCH_COUNT,
    EFFECTIVE_BATCH_SIZE,
    EQUITY_COUNT,
    FINAL_LR_FACTOR,
    INSTRUMENT_COUNT,
    MAX_EPOCHS,
    PATCH_INPUT_WIDTH,
    RUNTIME_PROFILES,
)
from brazil_rv.modeling.evaluate import _validate_run_checkpoint_identity
from brazil_rv.modeling.engine import (
    _filter_evaluation_rows,
    _optimizer_update,
    checkpoint_payload,
    compile_model,
    masked_huber_loss,
    train_one_epoch,
    validate_runtime_profile,
)
from brazil_rv.modeling.layers import MuonLinear
from brazil_rv.modeling.metrics import create_metric_table
from brazil_rv.modeling.train import _atomic_write_json
from brazil_rv.modeling.model import CrossAssetPatchITransformerV1
from brazil_rv.modeling.optim import (
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


def test_compile_configuration_uses_in_place_module_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_compile(module: nn.Module, **kwargs: object) -> None:
        assert isinstance(module, nn.Module)
        calls.append(kwargs)

    config = torch._functorch.config
    original_autocast = config.backward_pass_autocast
    monkeypatch.setattr(config, "backward_pass_autocast", original_autocast)
    monkeypatch.setattr(nn.Module, "compile", fake_compile)
    profile = RUNTIME_PROFILES["a10"]
    model = nn.Linear(2, 1)
    compile_model(model, profile)
    assert calls == [
        {
            "backend": "inductor",
            "mode": "reduce-overhead",
            "fullgraph": True,
            "dynamic": False,
        }
    ]
    assert config.backward_pass_autocast == "off"


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


def _matching_run_identity(feature_store: Path) -> dict[str, object]:
    return {
        "contract_version": "CROSS_ASSET_ITRANSFORMER_V1",
        "cloud_runtime_contract_version": ("CROSS_ASSET_ITRANSFORMER_CLOUD_RUNTIME_V1"),
        "model_variant": "full",
        "optimizer_variant": "hybrid",
        "seed": 11,
        "runtime_profile": "a10",
        "resolved_feature_store_path": str(feature_store),
        "git_commit_sha": "test-sha",
        "architecture_constants": {"d_model": 256},
    }


def test_evaluation_identity_accepts_match_and_different_evaluation_profile(
    tmp_path: Path,
) -> None:
    feature_store = tmp_path.resolve()
    manifest = _matching_run_identity(feature_store)
    checkpoint = dict(manifest)
    evaluation_profile = "a100"
    assert evaluation_profile != manifest["runtime_profile"]
    manifest["evaluation_runtime_profile"] = evaluation_profile
    _validate_run_checkpoint_identity(manifest, checkpoint, feature_store)


@pytest.mark.parametrize(
    "field",
    (
        "contract_version",
        "cloud_runtime_contract_version",
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
