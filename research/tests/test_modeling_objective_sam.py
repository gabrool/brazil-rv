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
    ADAMW_WEIGHT_DECAY,
    GH200_RUNTIME,
    SAM_RHOS,
    SOFT_RANK_TEMPERATURES,
)
from brazil_rv.modeling.engine import (
    checkpoint_payload,
    objective_metadata,
    run_effective_batch_update,
    sam_metadata,
    soft_spearman_loss,
)
from brazil_rv.modeling.model import build_neural_model
from brazil_rv.modeling.optim import build_optimizer


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
    assert objective_metadata(0.1) == {
        "name": "soft_spearman",
        "temperature": 0.1,
        "score_standardization": "masked_cross_sectional",
        "soft_rank": "pairwise_sigmoid",
        "aggregation": "equal_valid_cross_section_horizon",
        "reported_validation_metric": "hard_spearman",
    }
    with pytest.raises(ValueError):
        objective_metadata(0.3)
    targets = _rank_targets(4).bfloat16()
    predictions = targets.clone().requires_grad_()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = soft_spearman_loss(
            predictions, targets, torch.ones_like(targets, dtype=torch.bool), 0.1
        )
    loss.backward()
    assert loss.dtype == torch.float32
    assert torch.isfinite(predictions.grad).all()


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


def _adamw(model: nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
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
    ("optimizer_variant", "sam_rho"),
    [("adamw", None), ("sam_adamw", 0.02)],
)
def test_eager_and_compiled_updates_match(
    cpu_engine: None,
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
        0.1,
        sam_rho,
    )
    compiled_update = run_effective_batch_update(
        compiled_model,
        _microbatches(),
        compiled_optimizer,
        None,
        GH200_RUNTIME,
        optimizer_variant,
        0.1,
        sam_rho,
    )
    assert compiled_update["backward_passes"] == eager_update["backward_passes"]
    for eager, compiled in zip(
        eager_model.parameters(), compiled_base.parameters(), strict=True
    ):
        torch.testing.assert_close(compiled, eager, atol=2e-7, rtol=2e-7)


def test_eight_microbatches_match_concatenated_group_normalization(
    cpu_engine: None,
) -> None:
    batches = _microbatches()
    accumulated_model = _CrossSectionModel()
    reference_model = copy.deepcopy(accumulated_model)
    accumulated_optimizer = _adamw(accumulated_model)
    reference_optimizer = _adamw(reference_model)

    update = run_effective_batch_update(
        accumulated_model,
        batches,
        accumulated_optimizer,
        None,
        GH200_RUNTIME,
        "adamw",
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
    for actual, expected in zip(
        accumulated_model.parameters(), reference_model.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)


def test_incomplete_group_fails_before_forward(cpu_engine: None) -> None:
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
            0.1,
            None,
        )
    assert not model.dropout_masks


def test_sam_replays_same_batches_rng_and_counts(cpu_engine: None) -> None:
    torch.manual_seed(29)
    model = _CrossSectionModel(dropout=0.5).train()
    optimizer = _adamw(model)
    scheduler = _CountingScheduler()
    update = run_effective_batch_update(
        model,
        _microbatches(),
        optimizer,
        scheduler,
        GH200_RUNTIME,
        "sam_adamw",
        0.1,
        0.02,
    )
    assert update["backward_passes"] == 16
    assert scheduler.steps == 1
    assert len(model.dropout_masks) == 16
    for first, second in zip(
        model.dropout_masks[:8], model.dropout_masks[8:], strict=True
    ):
        torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_sam_matches_reference_l2_adamw_update(cpu_engine: None) -> None:
    batches = _microbatches()
    actual_model = _CrossSectionModel()
    reference_model = copy.deepcopy(actual_model)
    actual_optimizer = _adamw(actual_model)
    reference_optimizer = _adamw(reference_model)
    rho = 0.02

    run_effective_batch_update(
        actual_model,
        batches,
        actual_optimizer,
        None,
        GH200_RUNTIME,
        "sam_adamw",
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
            None,
            GH200_RUNTIME,
            "sam_adamw",
            0.1,
            0.02,
        )
    for parameter, expected in zip(model.parameters(), initial, strict=True):
        torch.testing.assert_close(parameter, expected, atol=0, rtol=0)


def test_rho_grid_metadata_cli_and_run_names() -> None:
    assert tuple(SAM_RHOS) == (0.0025, 0.005, 0.010, 0.020, 0.035, 0.050)
    assert sam_metadata("adamw", None) is None
    assert sam_metadata("sam_adamw", 0.02)["rho"] == 0.02
    for invalid in (0.0, -0.01, 0.03, 0.051):
        with pytest.raises(ValueError):
            sam_metadata("sam_adamw", invalid)
    with pytest.raises(ValueError):
        sam_metadata("adamw", 0.02)

    adamw = train.parse_args(
        [
            "--model",
            "tcn",
            "--optimizer",
            "adamw",
            "--soft-rank-temperature",
            "0.10",
            "--seed",
            "11",
        ]
    )
    assert adamw.sam_rho is None
    sam = train.parse_args(
        [
            "--model",
            "tcn",
            "--optimizer",
            "sam_adamw",
            "--soft-rank-temperature",
            "0.10",
            "--sam-rho",
            "0.020",
            "--seed",
            "11",
        ]
    )
    assert sam.sam_rho == 0.02
    with pytest.raises(SystemExit):
        train.parse_args(
            [
                "--model",
                "tcn",
                "--optimizer",
                "sam_adamw",
                "--soft-rank-temperature",
                "0.10",
                "--seed",
                "11",
            ]
        )

    created = datetime(2026, 1, 2, 3, 4, 5, 6789, tzinfo=timezone.utc)
    assert (
        train._run_directory_name("tcn", "adamw", 0.1, None, 11, created)
        == "tcn_adamw_tau0p10_seed11_20260102T030405006789Z"
    )
    assert (
        train._run_directory_name("tcn", "sam_adamw", 0.1, 0.01, 11, created)
        == "tcn_sam_adamw_rho0p010_tau0p10_seed11_20260102T030405006789Z"
    )


def test_checkpoint_round_trip_contains_resume_boundary_state(tmp_path: Path) -> None:
    model = build_neural_model("tcn")
    optimizer, _ = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    payload = checkpoint_payload(
        model,
        optimizer,
        scheduler,
        "tcn",
        "sam_adamw",
        0.1,
        0.02,
        11,
        3,
        0.12,
        tmp_path,
        "test-sha",
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    restored = build_neural_model("tcn")
    restored_optimizer, _ = build_optimizer(restored)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lambda _: 1.0
    )
    restored.load_state_dict(loaded["model_state_dict"])
    restored_optimizer.load_state_dict(loaded["optimizer_state_dict"])
    restored_scheduler.load_state_dict(loaded["scheduler_state_dict"])
    assert loaded["objective"] == objective_metadata(0.1)
    assert loaded["sam"] == sam_metadata("sam_adamw", 0.02)
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
