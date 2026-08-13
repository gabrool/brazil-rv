from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from brazil_rv.modeling.contract import (
    BASELINE_TCN_SETTINGS,
    RuntimeSettings,
    architecture_for_model,
)
from brazil_rv.modeling.engine import (
    _filter_evaluation_rows,
    checkpoint_payload,
    objective_loss,
    rank_huber_loss,
    run_effective_batch_update,
    soft_spearman_loss,
)
from brazil_rv.modeling.evaluate import load_current_neural_run
from brazil_rv.modeling.metrics import create_metric_table
from brazil_rv.modeling.model import build_neural_model
from brazil_rv.modeling.train import parse_args


class TinyRanker(nn.Module):
    model_name = "mlp"

    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(2, 3, bias=False)

    def forward(
        self, features: torch.Tensor, equity_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.linear(self.dropout(features)) * equity_mask[..., None]


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
    assert result["rng_replay_exact"] is True
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
    predictions = rng.normal(size=(4, 30, 3)).astype(np.float32)
    targets = rng.normal(size=(4, 30, 3)).astype(np.float32)
    returns = rng.normal(scale=0.01, size=(4, 30, 3)).astype(np.float32)
    mask = np.ones((4, 30, 3), dtype=bool)
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
        "sample_valid_mask": torch.tensor([True, True, False, False]),
        "sample_id": torch.tensor([4, 5, -1, -1]),
        "targets": torch.zeros(4, 2, 3),
        "raw_returns": torch.zeros(4, 2, 3),
        "label_mask": torch.ones(4, 2, 3, dtype=torch.bool),
        "date_idx": torch.tensor([1, 1, -1, -1]),
        "decision_idx": torch.tensor([0, 1, -1, -1]),
    }
    filtered = _filter_evaluation_rows(predictions, batch)
    assert filtered["predictions"].shape[0] == 2
    np.testing.assert_array_equal(filtered["sample_id"], [4, 5])


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
