from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from brazil_rv.modeling.engine import soft_spearman_loss
from brazil_rv.modeling.trajectory import ModelEMA
from brazil_rv.v2.config import INTRADAY_DAILY_FEATURES, ModelConfig
from brazil_rv.v2.contract import (
    DECISION_FEATURE_CONTRACT,
    DECISION_FEATURE_ALIGNMENT,
    FEATURE_AGE_CONTRACT,
    PRETRAIN_END,
    STORE_START,
    TARGET_NEUTRALIZATION_FEATURES,
)
from brazil_rv.v2.data import V2DailyDataset
from brazil_rv.v2.losses import (
    multi_horizon_loss,
    multi_horizon_loss_components,
    multi_horizon_loss_normalizers,
    score_persistence_penalty,
)
from brazil_rv.v2.model import DailyMultiHorizonModel
from brazil_rv.v2.train import (
    DatePairBatchSampler,
    PatienceTracker,
    build_optimizer,
    compile_forward,
    load_pretrain_handoff,
    load_stage_checkpoint,
    pretrain_internal_split,
    rank_average_ensemble,
    reshape_date_pair_batch,
    sam_accumulated_step,
    sam_step,
    train_stage,
    _common_primary_selection_score,
    _configure_inductor_compiler,
    _date_pair_microbatches,
    _input_static_identity,
    _loader_input_payload,
    _model_input_segments,
    _require_production_pair_sampler,
    _validate_tracked_stage_inputs,
)
from v2_store_fixtures import write_fixture_store as write_store


def test_external_location_is_not_part_of_model_input_identity() -> None:
    first = {
        "store": {
            "manifest_sha256": "a" * 64,
            "external_artifact_resolutions": [
                {
                    "recorded_path": r"D:\quant-data\sealed.npy",
                    "resolved_path": r"D:\quant-data\sealed.npy",
                    "bytes": 123,
                    "sha256": "b" * 64,
                    "override_file": None,
                }
            ],
        },
        "features": {"ordered": ["x"]},
        "lookback_sessions": 60,
    }
    relocated = json.loads(json.dumps(first))
    resolution = relocated["store"]["external_artifact_resolutions"][0]
    resolution["resolved_path"] = "/lambda/nfs/quant-data/sealed.npy"
    resolution["override_file"] = "/run-config/data_roots.json"

    assert _input_static_identity(first) == _input_static_identity(relocated)
    resolution["sha256"] = "c" * 64
    assert _input_static_identity(first) != _input_static_identity(relocated)


def test_multihead_objective_averages_each_horizon_separately() -> None:
    torch.manual_seed(3)
    scores = torch.randn(2, 5, 6)
    targets = torch.randn(2, 5, 6)
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[..., 5] = False
    components = multi_horizon_loss_components(scores, targets, mask)
    expected = torch.stack(
        [
            soft_spearman_loss(
                scores[..., head : head + 1],
                targets[..., head : head + 1],
                mask[..., head : head + 1],
            )
            for head in range(5)
        ]
    ).mean()
    assert torch.allclose(components["horizon"], expected)
    assert components["to_close"] == 0
    assert torch.allclose(components["total"], expected)


def test_five_heads_are_primary_and_to_close_is_explicitly_opt_in() -> None:
    torch.manual_seed(5)
    scores = torch.randn(2, 5, 6)
    targets = torch.randn(2, 5, 6)
    mask = torch.ones_like(scores, dtype=torch.bool)
    primary = multi_horizon_loss_components(
        scores[..., :5], targets[..., :5], mask[..., :5]
    )
    default_with_aux_present = multi_horizon_loss_components(scores, targets, mask)
    weighted = multi_horizon_loss_components(scores, targets, mask, to_close_weight=0.2)

    assert torch.allclose(primary["total"], default_with_aux_present["total"])
    assert torch.allclose(
        weighted["total"],
        weighted["horizon"] + 0.2 * weighted["to_close"],
    )
    with pytest.raises(ValueError, match="weighted to-close"):
        multi_horizon_loss_components(
            scores[..., :5],
            targets[..., :5],
            mask[..., :5],
            to_close_weight=0.2,
        )


def test_persistence_uses_population_zscores_and_score_mask() -> None:
    scores = torch.zeros(1, 2, 4, 6)
    scores[0, 0, :3, 0] = torch.tensor([-1.0, 0.0, 1.0])
    scores[0, 1, :3, 0] = torch.tensor([1.0, 0.0, -1.0])
    mask = torch.zeros(1, 2, 4, 5, dtype=torch.bool)
    mask[..., 0] = torch.tensor([True, True, True, False])
    penalty = score_persistence_penalty(scores, mask)
    assert penalty.item() == pytest.approx(4.0, abs=1e-4)
    scores[0, :, 3, 0] = 1e9
    assert score_persistence_penalty(scores, mask).item() == pytest.approx(
        penalty.item()
    )


def test_persistence_is_forced_to_float32_outside_autocast() -> None:
    scores = torch.randn(1, 2, 4, 6, dtype=torch.bfloat16)
    mask = torch.ones(1, 2, 4, 5, dtype=torch.bool)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        penalty = score_persistence_penalty(scores, mask)
    assert penalty.dtype == torch.float32


def test_microbatch_loss_and_gradients_match_full_effective_batch() -> None:
    generator = torch.Generator().manual_seed(71)
    full_scores = torch.randn(8, 2, 24, 6, generator=generator, requires_grad=True)
    micro_scores = full_scores.detach().clone().requires_grad_(True)
    targets = torch.randn(8, 2, 24, 6, generator=generator)
    mask = torch.rand(8, 2, 24, 6, generator=generator) > 0.18
    active = torch.rand(8, 2, 24, generator=generator) > 0.12
    normalizers = multi_horizon_loss_normalizers(mask, score_mask=active)
    full_loss = multi_horizon_loss(
        full_scores,
        targets,
        mask,
        score_mask=active,
        persistence_weight=0.1,
        to_close_weight=0.2,
        normalization_counts=normalizers,
    )
    full_loss.backward()

    micro_loss = sum(
        (
            multi_horizon_loss(
                micro_scores[start:stop],
                targets[start:stop],
                mask[start:stop],
                score_mask=active[start:stop],
                persistence_weight=0.1,
                to_close_weight=0.2,
                normalization_counts=normalizers,
            )
            for start, stop in ((0, 3), (3, 6), (6, 8))
        ),
        torch.zeros(()),
    )
    micro_loss.backward()

    assert torch.allclose(micro_loss, full_loss, rtol=2e-6, atol=2e-6)
    assert torch.allclose(micro_scores.grad, full_scores.grad, rtol=3e-5, atol=3e-6)


def test_selection_uses_one_supported_population_for_all_primary_heads() -> None:
    base = np.arange(24, dtype=np.float32)
    predictions = np.stack((base, base, base, base), axis=-1)[None]
    targets = predictions.copy()
    mask = np.ones_like(predictions, dtype=bool)
    active = np.ones((1, 24), dtype=bool)
    assert _common_primary_selection_score(
        predictions, targets, mask, active
    ) == pytest.approx(1.0)

    mask[0, :5, 3] = False
    with pytest.raises(ValueError, match="common four-head population"):
        _common_primary_selection_score(predictions, targets, mask, active)


def test_date_pair_sampler_keeps_adjacent_rows_and_is_deterministic() -> None:
    sampler = DatePairBatchSampler(range(7), pairs_per_batch=2, seed=11)
    first = list(sampler)
    second = list(sampler)
    assert first == second
    assert len(first) == 3
    for batch in first:
        for offset in range(0, len(batch), 2):
            assert batch[offset + 1] == batch[offset] + 1
    tensor = torch.arange(12).reshape(6, 2)
    assert reshape_date_pair_batch(tensor).shape == (3, 2, 2)


def test_microbatch_slicing_never_splits_a_date_pair() -> None:
    batch = {
        "slow_features": torch.arange(16).reshape(16, 1),
        "date_index": torch.arange(100, 116),
        "static": "kept",
    }
    pieces = _date_pair_microbatches(batch, 3)
    assert [piece["slow_features"].shape[0] for piece in pieces] == [6, 6, 4]
    assert [piece["date_index"].tolist() for piece in pieces] == [
        list(range(100, 106)),
        list(range(106, 112)),
        list(range(112, 116)),
    ]
    assert all(piece["static"] == "kept" for piece in pieces)


def test_production_pair_sampler_emits_only_exact_eight_pair_batches() -> None:
    sampler = DatePairBatchSampler(
        range(18), pairs_per_batch=8, seed=11, drop_last=True
    )
    batches = list(sampler)
    assert len(batches) == 2
    assert all(len(batch) == 16 for batch in batches)
    assert all(
        batch[offset + 1] == batch[offset] + 1
        for batch in batches
        for offset in range(0, len(batch), 2)
    )
    _require_production_pair_sampler(SimpleNamespace(batch_sampler=sampler))
    with pytest.raises(ValueError, match="exactly 8 pairs and drop_last=True"):
        _require_production_pair_sampler(
            SimpleNamespace(
                batch_sampler=DatePairBatchSampler(
                    range(18), pairs_per_batch=7, drop_last=True
                )
            )
        )
    with pytest.raises(ValueError, match="exactly 8 pairs and drop_last=True"):
        _require_production_pair_sampler(
            SimpleNamespace(
                batch_sampler=DatePairBatchSampler(
                    range(18), pairs_per_batch=8, drop_last=False
                )
            )
        )


def test_time_decay_sampler_is_epoch_deterministic() -> None:
    left = DatePairBatchSampler(
        range(20), seed=47, time_decay_half_life=756.0, pairs_per_batch=4
    )
    right = DatePairBatchSampler(
        range(20), seed=47, time_decay_half_life=756.0, pairs_per_batch=4
    )
    left.set_epoch(2)
    right.set_epoch(2)
    assert list(left) == list(right)


def test_stage_j_requires_the_frozen_time_decay_and_other_stages_reject_it(
    tmp_path,
) -> None:
    base = ModelConfig(slow_feature_count=32, compile_forward=False)
    with pytest.raises(
        ValueError, match="stage J requires time_decay_half_life_sessions=756.0"
    ):
        train_stage(
            stage="J",
            seed=11,
            fold="F1",
            train_loader=[],
            selection_loader=[],
            output_dir=tmp_path / "j_without_decay",
            model_config=base,
            selection_parity=None,
            maximum_epochs=1,
        )
    with pytest.raises(
        ValueError, match="stage F requires time_decay_half_life_sessions=None"
    ):
        train_stage(
            stage="F",
            seed=11,
            fold="F1",
            train_loader=[],
            selection_loader=[],
            output_dir=tmp_path / "f_with_decay",
            model_config=ModelConfig(
                slow_feature_count=32,
                compile_forward=False,
                time_decay_half_life_sessions=756.0,
            ),
            selection_parity=None,
            maximum_epochs=1,
        )


def test_date_pair_sampler_never_crosses_a_window_gap() -> None:
    sampler = DatePairBatchSampler(
        (4, 5, 9, 10), session_indices=(20, 21, 40, 41), pairs_per_batch=8
    )
    assert list(sampler) == [[0, 1, 2, 3]]
    assert list(DatePairBatchSampler((20, 21, 40, 41), pairs_per_batch=8)) == [
        [0, 1, 2, 3]
    ]


def test_pretrain_internal_holdout_is_last_ten_percent_after_embargo() -> None:
    fit, embargo, selection = pretrain_internal_split(np.arange(1_000))
    assert fit.tolist() == list(range(830))
    assert embargo.tolist() == list(range(830, 900))
    assert selection.tolist() == list(range(900, 1_000))


def test_rank_average_ensemble_is_per_group_and_tie_aware() -> None:
    left = np.array([[[3.0], [1.0], [1.0], [9.0]]])
    right = np.array([[[0.0], [4.0], [2.0], [8.0]]])
    mask = np.array([[True, True, True, False]])
    actual = rank_average_ensemble((left, right), mask)
    assert actual[0, :, 0].tolist() == [1.0, 1.25, 0.75, 0.0]


def test_optimizer_routes_pretrained_parameters_at_point_three_lr() -> None:
    model = DailyMultiHorizonModel(ModelConfig(slow_feature_count=32))
    pretrained = frozenset(
        f"fast_encoder.{name}" for name, _ in model.fast_encoder.named_parameters()
    )
    optimizer = build_optimizer(model, pretrained_parameter_names=pretrained)
    grouped = {
        (group["pretrained"], group["weight_decay"]): group["lr"]
        for group in optimizer.param_groups
    }
    assert grouped[(True, 0.01)] == pytest.approx(9e-5)
    assert grouped[(False, 0.01)] == pytest.approx(3e-4)
    decay_ids = {
        id(parameter)
        for group in optimizer.param_groups
        if group["weight_decay"] == 0.01
        for parameter in group["params"]
    }
    assert id(model.slow_input_norm.weight) in decay_ids
    assert id(model.absent_state) not in decay_ids


def test_actual_patience_restores_best_raw_state() -> None:
    model = nn.Linear(1, 1, bias=False)
    tracker = PatienceTracker(patience=3, maximum_epochs=20)
    stopped = False
    for epoch, score in enumerate((1.0, 0.9, 0.8, 0.7), start=1):
        model.weight.data.fill_(epoch)
        stopped = tracker.update(epoch, score, model)
    assert stopped
    assert tracker.selected_epoch == 1
    assert tracker.stopped_epoch == 4
    tracker.restore(model)
    assert model.weight.item() == 1.0


def test_sam_updates_model_and_ema() -> None:
    torch.manual_seed(13)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    ema = ModelEMA(model, 0.995)
    inputs = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
    targets = torch.tensor([[1.0], [-0.5]])
    before = model.weight.detach().clone()

    def closure() -> torch.Tensor:
        return (model(inputs) - targets).square().mean()

    result = sam_step(model, optimizer, closure, ema=ema)
    assert np.isfinite(result.first_gradient_norm)
    assert not torch.equal(before, model.weight)
    assert not torch.equal(ema.shadow["weight"], before)


def test_accumulated_sam_matches_full_batch_update_and_advances_once() -> None:
    torch.manual_seed(83)
    full_model = nn.Linear(3, 2)
    accumulated_model = nn.Linear(3, 2)
    accumulated_model.load_state_dict(full_model.state_dict())
    inputs = torch.randn(16, 3)
    targets = torch.randn(16, 2)
    full_optimizer = torch.optim.AdamW(full_model.parameters(), lr=1e-3)
    accumulated_optimizer = torch.optim.AdamW(accumulated_model.parameters(), lr=1e-3)
    full_scheduler = torch.optim.lr_scheduler.StepLR(full_optimizer, step_size=1)
    accumulated_scheduler = torch.optim.lr_scheduler.StepLR(
        accumulated_optimizer, step_size=1
    )
    full_ema = ModelEMA(full_model, 0.995)
    accumulated_ema = ModelEMA(accumulated_model, 0.995)

    def full_closure() -> torch.Tensor:
        return (full_model(inputs) - targets).square().mean()

    denominator = float(targets.numel())
    closures = tuple(
        (
            lambda start=start, stop=stop: (
                (accumulated_model(inputs[start:stop]) - targets[start:stop])
                .square()
                .sum()
                / denominator
            )
        )
        for start, stop in ((0, 4), (4, 10), (10, 16))
    )
    full_result = sam_step(
        full_model,
        full_optimizer,
        full_closure,
        scheduler=full_scheduler,
        ema=full_ema,
    )
    accumulated_result = sam_accumulated_step(
        accumulated_model,
        accumulated_optimizer,
        closures,
        scheduler=accumulated_scheduler,
        ema=accumulated_ema,
    )

    for full, accumulated in zip(
        full_model.parameters(), accumulated_model.parameters(), strict=True
    ):
        assert torch.allclose(full, accumulated, rtol=1e-6, atol=1e-7)
    assert accumulated_result.first_loss == pytest.approx(
        full_result.first_loss, rel=1e-6, abs=1e-7
    )
    assert accumulated_scheduler.last_epoch == full_scheduler.last_epoch == 1
    for name in full_ema.shadow:
        assert torch.allclose(
            full_ema.shadow[name], accumulated_ema.shadow[name], rtol=1e-6, atol=1e-7
        )


@pytest.mark.parametrize("failing_pass", [1, 2])
def test_sam_rejects_each_nonfinite_training_loss(failing_pass: int) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    calls = 0

    def closure() -> torch.Tensor:
        nonlocal calls
        calls += 1
        loss = model.weight.square().sum()
        return loss * torch.nan if calls == failing_pass else loss

    with pytest.raises(FloatingPointError, match="training loss is non-finite"):
        sam_step(model, optimizer, closure)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_fullgraph_compile_captures_gru_forward() -> None:
    config = ModelConfig(slow_feature_count=32, slow_lookback=20)
    model = DailyMultiHorizonModel(config).eval()
    compiled = compile_forward(model, backend="eager", mode=None)
    current_features = torch.randn(2, 3, config.current_feature_count)
    current_feature_mask = torch.ones_like(current_features, dtype=torch.bool)
    slow_feature_age_sessions = torch.zeros(2, 3, 20, 32)
    current_feature_age_sessions = torch.zeros_like(current_features)
    scores = compiled(
        torch.randn(2, 3, 20, 32),
        torch.ones(2, 3, 20, 32, dtype=torch.bool),
        torch.ones(2, 3, 20, dtype=torch.bool),
        torch.ones(2, 3, dtype=torch.bool),
        current_features=current_features,
        current_feature_mask=current_feature_mask,
        slow_feature_age_sessions=slow_feature_age_sessions,
        current_feature_age_sessions=current_feature_age_sessions,
    )
    assert scores.shape == (2, 3, 6)
    fast_values = torch.randn(2, 2, 5, 7)
    fast_valid = torch.ones_like(fast_values, dtype=torch.bool)
    fast_mask = torch.ones(2, 2, 5, dtype=torch.bool)
    fast_mask[1, 1] = False
    fast_valid[1, 1] = False
    fast_values[1, 1] = 0.0
    fast_scores = compiled(
        torch.randn(2, 3, 20, 32),
        torch.ones(2, 3, 20, 32, dtype=torch.bool),
        torch.ones(2, 3, 20, dtype=torch.bool),
        torch.ones(2, 3, dtype=torch.bool),
        fast_patch_values=fast_values,
        fast_patch_valid=fast_valid,
        fast_patch_mask=fast_mask,
        fast_name_index=torch.tensor([[0, 2], [1, -1]]),
        fast_state_position=torch.tensor([[5, 5], [5, 0]]),
        fast_present=torch.tensor([[True, False, True], [False, True, False]]),
        current_features=current_features,
        current_feature_mask=current_feature_mask,
        slow_feature_age_sessions=slow_feature_age_sessions,
        current_feature_age_sessions=current_feature_age_sessions,
    )
    assert fast_scores.shape == (2, 3, 6)


def test_arm64_inductor_binds_the_gh200_capable_compiler(monkeypatch) -> None:
    original = torch._inductor.config.cpp.cxx
    monkeypatch.setattr("brazil_rv.v2.train.platform.machine", lambda: "aarch64")
    monkeypatch.setattr(
        "brazil_rv.v2.train.shutil.which",
        lambda name: "/usr/bin/g++-12" if name == "g++-12" else None,
    )
    try:
        _configure_inductor_compiler()
        assert torch._inductor.config.cpp.cxx == ("/usr/bin/g++-12",)
    finally:
        torch._inductor.config.cpp.cxx = original


def test_arm64_inductor_fails_before_compile_without_gxx12(monkeypatch) -> None:
    monkeypatch.setattr("brazil_rv.v2.train.platform.machine", lambda: "aarch64")
    monkeypatch.setattr("brazil_rv.v2.train.shutil.which", lambda name: None)

    with pytest.raises(RuntimeError, match=r"requires g\+\+-12"):
        _configure_inductor_compiler()


def test_fullgraph_compile_reuses_one_graph_across_date_batch_widths() -> None:
    config = ModelConfig(slow_feature_count=32, slow_lookback=20)
    model = DailyMultiHorizonModel(config).eval()
    compiled = compile_forward(model, backend="eager", mode=None)
    before = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])

    for batch_size in (16, 1):
        current = torch.randn(batch_size, 3, config.current_feature_count)
        compiled(
            torch.randn(batch_size, 3, 20, 32),
            torch.ones(batch_size, 3, 20, 32, dtype=torch.bool),
            torch.ones(batch_size, 3, 20, dtype=torch.bool),
            torch.ones(batch_size, 3, dtype=torch.bool),
            current_features=current,
            current_feature_mask=torch.ones_like(current, dtype=torch.bool),
            slow_feature_age_sessions=torch.zeros(batch_size, 3, 20, 32),
            current_feature_age_sessions=torch.zeros_like(current),
        )

    assert int(torch._dynamo.utils.counters["stats"]["unique_graphs"]) - before == 1


def test_compiler_keeps_one_specialization_per_required_module_mode() -> None:
    class ModeSensitive(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dropout = nn.Dropout(0.1)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.dropout(values)

    model = ModeSensitive()
    compiled = compile_forward(model, backend="eager", mode=None)
    before = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
    model.train()
    compiled(torch.ones(16, 3))
    model.eval()
    compiled(torch.ones(1, 3))

    assert int(torch._dynamo.utils.counters["stats"]["unique_graphs"]) - before == 2


def test_fixed_fast_padding_matches_eager_and_compiled_cpu() -> None:
    config = ModelConfig(slow_feature_count=32, slow_lookback=20)
    model = DailyMultiHorizonModel(config).eval()
    batch_size, names, fast_width = 2, 3, 16
    slow = torch.randn(batch_size, names, 20, 32)
    slow_mask = torch.ones_like(slow, dtype=torch.bool)
    timestep_mask = torch.ones(batch_size, names, 20, dtype=torch.bool)
    active = torch.ones(batch_size, names, dtype=torch.bool)
    current = torch.randn(batch_size, names, config.current_feature_count)
    current_mask = torch.ones_like(current, dtype=torch.bool)
    fast_values = torch.zeros(batch_size, fast_width, 5, 7)
    fast_valid = torch.zeros_like(fast_values, dtype=torch.bool)
    fast_mask = torch.zeros(batch_size, fast_width, 5, dtype=torch.bool)
    fast_name_index = torch.full((batch_size, fast_width), -1, dtype=torch.int64)
    fast_state_position = torch.zeros(batch_size, fast_width, dtype=torch.int64)
    fast_present = torch.zeros(batch_size, names, dtype=torch.bool)
    fast_values[:, :2] = torch.randn(batch_size, 2, 5, 7)
    fast_valid[:, :2] = True
    fast_mask[:, :2] = True
    fast_name_index[:, :2] = torch.tensor([[0, 2], [1, 2]])
    fast_state_position[:, :2] = 5
    fast_present[0, [0, 2]] = True
    fast_present[1, [1, 2]] = True
    kwargs = {
        "fast_patch_values": fast_values,
        "fast_patch_valid": fast_valid,
        "fast_patch_mask": fast_mask,
        "fast_name_index": fast_name_index,
        "fast_state_position": fast_state_position,
        "fast_present": fast_present,
        "current_features": current,
        "current_feature_mask": current_mask,
        "slow_feature_age_sessions": torch.zeros_like(slow),
        "current_feature_age_sessions": torch.zeros_like(current),
    }
    with torch.no_grad():
        eager = model(slow, slow_mask, timestep_mask, active, **kwargs)
        unpadded_kwargs = {
            **kwargs,
            "fast_patch_values": fast_values[:, :2],
            "fast_patch_valid": fast_valid[:, :2],
            "fast_patch_mask": fast_mask[:, :2],
            "fast_name_index": fast_name_index[:, :2],
            "fast_state_position": fast_state_position[:, :2],
        }
        unpadded = model(slow, slow_mask, timestep_mask, active, **unpadded_kwargs)
        compiled = compile_forward(model, backend="eager", mode=None)(
            slow, slow_mask, timestep_mask, active, **kwargs
        )
    assert eager.shape == (batch_size, names, 6)
    assert torch.isfinite(eager.square().mean())
    torch.testing.assert_close(eager, unpadded, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(compiled, eager)
    targets = torch.rand(batch_size, names, 5)
    target_mask = torch.ones_like(targets, dtype=torch.bool)
    padded_loss = multi_horizon_loss(eager[..., :5], targets, target_mask)
    unpadded_loss = multi_horizon_loss(unpadded[..., :5], targets, target_mask)
    torch.testing.assert_close(padded_loss, unpadded_loss, rtol=1e-6, atol=1e-7)


def _tracked_pretrain_loaders(tmp_path):
    calendar = np.arange(
        np.datetime64(STORE_START),
        np.datetime64(PRETRAIN_END) + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )
    dates = calendar[np.is_busday(calendar)]
    fit, _, selection = pretrain_internal_split(np.arange(dates.size))
    generator = np.random.default_rng(19)
    name_count = 24
    slow = generator.standard_normal((dates.size, name_count, 32)).astype(np.float32)
    targets = generator.standard_normal((dates.size, name_count, 5)).astype(np.float32)
    intraday = generator.standard_normal(
        (dates.size, name_count, len(INTRADAY_DAILY_FEATURES))
    ).astype(np.float32)
    store = write_store(
        tmp_path / "tracked_store",
        dates=dates,
        isins=[f"BRTEST{index:02d}NOR1" for index in range(name_count)],
        arrays={
            "slow_values": slow,
            "slow_valid": np.ones_like(slow, dtype=np.bool_),
            "slow_age_sessions": np.zeros_like(slow, dtype=np.float32),
            "slow_timestep_valid": np.ones((dates.size, name_count), dtype=np.bool_),
            "intraday_values": intraday,
            "intraday_valid": np.ones_like(intraday, dtype=np.bool_),
            "intraday_age_sessions": np.zeros_like(intraday, dtype=np.float32),
            "active": np.ones((dates.size, name_count), dtype=np.bool_),
            "target_primary": targets,
            "target_valid": np.ones_like(targets, dtype=np.bool_),
            "target_shareholder_midrank": targets,
            "target_shareholder_simple_return": targets,
            "target_shareholder_valid": np.ones_like(targets, dtype=np.bool_),
            "target_scale_sigma": np.ones((dates.size, name_count), dtype=np.float32),
        },
        feature_names={
            "slow": [
                *TARGET_NEUTRALIZATION_FEATURES,
                *(f"slow_{index}" for index in range(3, 32)),
            ],
            "intraday": list(INTRADAY_DAILY_FEATURES),
        },
        metadata={
            "feature_age_contract": dict(FEATURE_AGE_CONTRACT),
            "slow_entry_alignment": dict(DECISION_FEATURE_CONTRACT),
        },
    )
    train_dataset = V2DailyDataset(
        store,
        fit[-9:],
        stage="pretrain",
        lookback=20,
        purpose="training",
    )
    selection_dataset = V2DailyDataset(
        store,
        selection[:6],
        stage="pretrain",
        lookback=20,
        purpose="selection",
    )

    def train_loader():
        return DataLoader(
            train_dataset,
            batch_sampler=DatePairBatchSampler(
                train_dataset.date_indices,
                pairs_per_batch=8,
                seed=19,
                drop_last=True,
            ),
            num_workers=0,
        )

    selection_loader = DataLoader(
        selection_dataset,
        batch_size=len(selection_dataset),
        shuffle=False,
        num_workers=0,
    )
    return train_loader, selection_loader


def test_model_input_identity_requires_canonical_row_and_feature_age_contract(
    tmp_path,
) -> None:
    train_loader_factory, _ = _tracked_pretrain_loaders(tmp_path)
    loader = train_loader_factory()
    payload = _loader_input_payload(loader)
    assert payload is not None
    features = payload["features"]
    assert payload["target"] == {
        "value_array": "target_primary_neutral",
        "validity_array": "target_primary_neutral_valid",
    }
    assert features["decision_sample_schema"] == "BRAZIL_RV_V2_DECISION_SAMPLE_V2"
    assert features["decision_feature_contract"] == DECISION_FEATURE_CONTRACT
    assert features["feature_age_contract"] == FEATURE_AGE_CONTRACT

    changed = json.loads(json.dumps(payload))
    changed["features"]["feature_age_contract"]["unit"] = "calendar_days"
    assert _input_static_identity(payload) != _input_static_identity(changed)

    metadata = loader.dataset.store.manifest["metadata"]
    age_contract = metadata["feature_age_contract"]
    metadata["feature_age_contract"] = {**age_contract, "unit": "calendar_days"}
    with pytest.raises(ValueError, match="feature-age contract"):
        _loader_input_payload(loader)
    metadata["feature_age_contract"] = age_contract

    decision_contract = metadata["slow_entry_alignment"]
    metadata["slow_entry_alignment"] = {
        **decision_contract,
        "consumer_side_shift": True,
    }
    with pytest.raises(ValueError, match="decision-feature alignment"):
        _loader_input_payload(loader)


def test_stage_runner_archives_patience_ema_and_handoff(tmp_path) -> None:
    config = ModelConfig(
        slow_feature_count=32,
        slow_lookback=20,
        compile_forward=False,
    )
    train_loader_factory, selection_loader = _tracked_pretrain_loaders(tmp_path)
    train_loader = train_loader_factory()
    output_dir = tmp_path / "stage_p"
    output_dir.mkdir()
    (output_dir / "launcher.stdout.log").write_text("", encoding="utf-8")
    (output_dir / "launcher.stderr.log").write_text("", encoding="utf-8")
    result = train_stage(
        stage="P",
        seed=29,
        fold="pretrain_internal",
        train_loader=train_loader,
        selection_loader=selection_loader,
        output_dir=output_dir,
        model_config=config,
        maximum_epochs=1,
        patience=1,
        device=torch.device("cpu"),
    )
    assert result.raw_patience_checkpoint.is_file()
    assert result.final_ema_checkpoint.is_file()
    assert result.history_path.is_file()
    assert result.manifest_path.is_file()
    assert train_loader.batch_sampler.epoch == 0
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["seed"] == 29
    assert manifest["fold"] == "pretrain_internal"
    assert manifest["compiled_graph_count"] == 0
    assert manifest["compiled_graphs"] == {
        "training": 0,
        "selection": 0,
        "total": 0,
    }
    assert manifest["official_validation_accessed"] is False
    assert manifest["test_accessed"] is False
    assert "allow_untracked_test_loaders" not in manifest
    assert manifest["fast_initialization_provenance"] == {
        "mode": "fresh",
        "contaminated": False,
        "explicitly_allowed": False,
        "checkpoint_sha256": None,
    }
    assert manifest["transfer_chronology_clean"] is True
    assert len(manifest["feature_schema_sha256"]) == 64
    assert manifest["checkpoint_input_contract"]["training"] is not None
    assert manifest["checkpoint_input_contract"]["selection"] is not None

    repeat_loader = train_loader_factory()
    repeated = train_stage(
        stage="P",
        seed=29,
        fold="pretrain_internal",
        train_loader=repeat_loader,
        selection_loader=selection_loader,
        output_dir=tmp_path / "stage_p_repeat",
        model_config=config,
        maximum_epochs=1,
        patience=1,
        device=torch.device("cpu"),
    )
    assert result.history_path.read_bytes() == repeated.history_path.read_bytes()
    raw_payload = torch.load(
        result.raw_patience_checkpoint, map_location="cpu", weights_only=False
    )
    first_state = raw_payload["model_state_dict"]
    assert (
        raw_payload["fast_initialization_provenance"]
        == manifest["fast_initialization_provenance"]
    )
    assert raw_payload["transfer_chronology_clean"] is True
    assert raw_payload["feature_schema_sha256"] == manifest["feature_schema_sha256"]
    repeated_state = torch.load(
        repeated.raw_patience_checkpoint, map_location="cpu", weights_only=False
    )["model_state_dict"]
    assert first_state.keys() == repeated_state.keys()
    assert all(
        torch.equal(first_state[name], repeated_state[name]) for name in first_state
    )

    loaded = DailyMultiHorizonModel(config)
    checkpoint_sha256 = load_stage_checkpoint(loaded, result.raw_patience_checkpoint)
    assert checkpoint_sha256 == manifest["artifacts"]["raw_patience.pt"]
    assert all(
        torch.equal(loaded.state_dict()[name], first_state[name])
        for name in first_state
    )

    fine_tune = DailyMultiHorizonModel(config)
    fast_before = {
        name: value.clone()
        for name, value in fine_tune.fast_encoder.state_dict().items()
    }
    initialized = load_pretrain_handoff(
        fine_tune,
        result.raw_patience_checkpoint,
        expected_sha256=manifest["artifacts"]["raw_patience.pt"],
        expected_seed=29,
    )
    assert initialized
    assert all(
        torch.equal(value, fast_before[name])
        for name, value in fine_tune.fast_encoder.state_dict().items()
    )
    ablated = DailyMultiHorizonModel(replace(config, disable_fast_stream=True))
    transferred = load_pretrain_handoff(
        ablated,
        result.raw_patience_checkpoint,
        expected_sha256=manifest["artifacts"]["raw_patience.pt"],
        expected_seed=29,
    )
    assert transferred == initialized
    assert all(
        torch.equal(ablated.state_dict()[name], fine_tune.state_dict()[name])
        for name in transferred
    )
    with pytest.raises(ValueError, match="model contract differs"):
        load_pretrain_handoff(
            DailyMultiHorizonModel(
                replace(config, disable_fast_stream=True, dropout=0.2)
            ),
            result.raw_patience_checkpoint,
            expected_sha256=manifest["artifacts"]["raw_patience.pt"],
            expected_seed=29,
        )

    base_payload = torch.load(
        result.raw_patience_checkpoint, map_location="cpu", weights_only=False
    )
    for index, (field, value) in enumerate(
        (
            ("schema", "BRAZIL_RV_V2_FINAL_EMA_0995_V2"),
            ("stage", "F"),
            ("seed", 47),
            ("fold", ""),
        )
    ):
        invalid = dict(base_payload)
        invalid[field] = value
        path = tmp_path / f"invalid_handoff_{index}.pt"
        torch.save(invalid, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with pytest.raises(
            ValueError, match="handoff schema, stage, optional seed, or fold"
        ):
            load_pretrain_handoff(
                DailyMultiHorizonModel(config),
                path,
                expected_sha256=digest,
                expected_seed=29,
            )
    invalid_contract = dict(base_payload)
    invalid_contract["input_contract"] = dict(base_payload["input_contract"])
    invalid_contract["input_contract"]["schema"] = "tampered"
    invalid_contract_path = tmp_path / "invalid_handoff_contract.pt"
    torch.save(invalid_contract, invalid_contract_path)
    with pytest.raises(ValueError, match="input contract hash"):
        load_pretrain_handoff(
            DailyMultiHorizonModel(config),
            invalid_contract_path,
            expected_sha256=hashlib.sha256(
                invalid_contract_path.read_bytes()
            ).hexdigest(),
            expected_seed=29,
        )


def test_stage_runner_rejects_untracked_production_loaders(tmp_path) -> None:
    with pytest.raises(ValueError, match="authorized access ledgers"):
        train_stage(
            stage="P",
            seed=29,
            fold="pretrain_internal",
            train_loader=[],
            selection_loader=[],
            output_dir=tmp_path / "untracked",
            model_config=ModelConfig(
                slow_feature_count=32,
                slow_lookback=20,
                compile_forward=False,
            ),
            maximum_epochs=1,
            patience=1,
            device=torch.device("cpu"),
        )


def _tracked_input(
    *,
    first_index: int,
    last_index: int,
    first_date: str,
    last_date: str,
    alignment: str,
    canonical_splits: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": "BRAZIL_RV_V2_MODEL_INPUT_V2",
        "store": {
            "schema": "BRAZIL_RV_V2_DAILY_STORE_V3",
            "manifest_sha256": "a" * 64,
            "axes": {"date_count": 1_000, "isin_count": 4},
            "fast_identity": {},
        },
        "features": {
            "ordered_slow_and_sidecar_names": [f"slow_{i}" for i in range(32)],
            "enabled_sidecar_groups": [],
            "ordered_sidecar_names": {},
            "ordered_intraday_names": list(INTRADAY_DAILY_FEATURES),
        },
        "lookback_sessions": 20,
        "entry_alignment": alignment,
        "canonical_splits": canonical_splits,
        "dates": {
            "first_index": first_index,
            "last_index": last_index,
            "first_date": first_date,
            "last_date": last_date,
        },
    }


def test_stage_input_contract_rejects_overlap_and_wrong_f_purge() -> None:
    canonical = {
        "F1": {
            "fit": {"first_index": 0, "last_index": 9},
            "purge_before": {"first_index": 10, "last_index": 19},
            "selection": {"first_index": 20, "last_index": 74},
            "purge_after": {"first_index": 75, "last_index": 84},
            "evaluation": {"first_index": 85, "last_index": 120},
        }
    }
    training = _tracked_input(
        first_index=0,
        last_index=9,
        first_date="2021-08-16",
        last_date="2023-03-17",
        alignment=DECISION_FEATURE_ALIGNMENT,
        canonical_splits=canonical,
    )
    valid_selection = _tracked_input(
        first_index=20,
        last_index=25,
        first_date="2023-03-20",
        last_date="2023-03-27",
        alignment=DECISION_FEATURE_ALIGNMENT,
        canonical_splits=canonical,
    )
    config = ModelConfig(slow_feature_count=32, slow_lookback=20)
    _validate_tracked_stage_inputs("F", "F1", config, training, valid_selection)
    overlap = dict(valid_selection)
    overlap["dates"] = {
        **valid_selection["dates"],
        "first_index": 9,
    }
    with pytest.raises(ValueError, match="ordered and disjoint"):
        _validate_tracked_stage_inputs("F", "F1", config, training, overlap)
    short_embargo = dict(valid_selection)
    short_embargo["dates"] = {
        **valid_selection["dates"],
        "first_index": 19,
    }
    with pytest.raises(ValueError, match="10-session purge"):
        _validate_tracked_stage_inputs("F", "F1", config, training, short_embargo)


def test_stage_input_contract_rejects_wrong_p_embargo_and_boundaries() -> None:
    canonical = {
        "P": {
            "fit": {"first_index": 0, "last_index": 99},
            "selection": {"first_index": 170, "last_index": 199},
            "embargo_sessions": 70,
            "selection_fraction": 0.10,
        }
    }
    training = _tracked_input(
        first_index=0,
        last_index=99,
        first_date="2010-01-04",
        last_date="2020-12-31",
        alignment=DECISION_FEATURE_ALIGNMENT,
        canonical_splits=canonical,
    )
    selection = _tracked_input(
        first_index=169,
        last_index=199,
        first_date="2021-01-04",
        last_date="2021-07-30",
        alignment=DECISION_FEATURE_ALIGNMENT,
        canonical_splits=canonical,
    )
    config = ModelConfig(slow_feature_count=32, slow_lookback=20)
    with pytest.raises(ValueError, match="70-session embargo"):
        _validate_tracked_stage_inputs(
            "P", "pretrain_internal", config, training, selection
        )
    selection["dates"] = {
        **selection["dates"],
        "first_index": 169,
    }
    training["dates"] = {
        **training["dates"],
        "last_index": 98,
    }
    with pytest.raises(ValueError, match="canonical P boundaries"):
        _validate_tracked_stage_inputs(
            "P", "pretrain_internal", config, training, selection
        )


def _joint_segment(
    name: str,
    alignment: str,
    first_index: int,
    last_index: int,
    first_date: str,
    last_date: str,
) -> dict[str, object]:
    return {
        "name": name,
        "entry_alignment": alignment,
        "indices_sha256": "1" * 64,
        "identity_sha256": "2" * 64,
        "count": last_index - first_index + 1,
        "first_index": first_index,
        "last_index": last_index,
        "first_date": first_date,
        "last_date": last_date,
    }


def test_joint_input_contract_records_and_enforces_ordered_p_f_segments() -> None:
    axis = np.arange(
        np.datetime64("2010-01-04"),
        np.datetime64("2021-08-20"),
        dtype="datetime64[D]",
    )
    pretrain = np.flatnonzero(axis <= np.datetime64("2021-07-30"))
    finetune = np.flatnonzero(axis >= np.datetime64("2021-08-16"))
    segments = _model_input_segments(
        axis, np.concatenate((pretrain, finetune)), "joint"
    )
    assert [segment["name"] for segment in segments] == ["P", "F"]
    assert [segment["entry_alignment"] for segment in segments] == [
        DECISION_FEATURE_ALIGNMENT,
        DECISION_FEATURE_ALIGNMENT,
    ]
    assert segments[0]["indices_sha256"] != segments[1]["indices_sha256"]

    canonical = {
        "P": {
            "fit": {"first_index": 0, "last_index": 69},
            "selection": {"first_index": 90, "last_index": 99},
            "embargo_sessions": 70,
            "selection_fraction": 0.10,
        },
        "F1": {
            "fit": {
                "first_index": 110,
                "last_index": 119,
                "first_date": "2021-08-16",
                "last_date": "2023-03-17",
                "count": 10,
            },
            "purge_before": {"first_index": 120, "last_index": 129},
            "selection": {
                "first_index": 130,
                "last_index": 184,
                "first_date": "2023-03-20",
                "last_date": "2023-06-02",
                "count": 55,
            },
            "purge_after": {"first_index": 185, "last_index": 194},
            "evaluation": {"first_index": 195, "last_index": 200},
        },
    }
    training = _tracked_input(
        first_index=0,
        last_index=119,
        first_date="2010-01-04",
        last_date="2023-03-17",
        alignment=DECISION_FEATURE_ALIGNMENT,
        canonical_splits=canonical,
    )
    training["segments"] = [
        _joint_segment(
            "P", DECISION_FEATURE_ALIGNMENT, 0, 99, "2010-01-04", "2021-07-30"
        ),
        _joint_segment(
            "F",
            DECISION_FEATURE_ALIGNMENT,
            110,
            119,
            "2021-08-16",
            "2023-03-17",
        ),
    ]
    selection = _tracked_input(
        first_index=130,
        last_index=184,
        first_date="2023-03-20",
        last_date="2023-06-02",
        alignment=DECISION_FEATURE_ALIGNMENT,
        canonical_splits=canonical,
    )
    selection["segments"] = [
        _joint_segment(
            "F",
            DECISION_FEATURE_ALIGNMENT,
            130,
            184,
            "2023-03-20",
            "2023-06-02",
        )
    ]
    config = ModelConfig(slow_feature_count=32, slow_lookback=20)
    _validate_tracked_stage_inputs("J", "F1", config, training, selection)

    training["segments"] = list(reversed(training["segments"]))
    with pytest.raises(ValueError, match="ordered P/F training"):
        _validate_tracked_stage_inputs("J", "F1", config, training, selection)
