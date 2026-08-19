from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from brazil_rv.modeling.contract import MAX_EPOCHS, RuntimeSettings
from brazil_rv.modeling.engine import (
    _soft_spearman_loss_sum,
    checkpoint_payload,
    compile_training_objective,
    objective_metadata,
    run_effective_batch_update,
    soft_spearman_loss,
)
from brazil_rv.modeling.model import SharedCausalTCN
from brazil_rv.modeling.provenance import model_metadata, repository_commit
from brazil_rv.modeling.train import parse_args
from brazil_rv.modeling.trajectory import (
    ModelEMA,
    average_state_dicts,
    retrospective_best_epoch,
    simulate_patience3,
)


def _rank_case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predictions = torch.tensor(
        [
            [[0.1, 0.3], [0.2, 0.1], [0.4, 0.2], [0.3, 0.4]],
            [[0.4, 0.2], [0.1, 0.4], [0.2, 0.3], [0.3, 0.1]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [
            [[-0.75, -0.25], [-0.25, -0.75], [0.75, 0.25], [0.25, 0.75]],
            [[0.75, -0.25], [-0.75, 0.75], [-0.25, 0.25], [0.25, -0.75]],
        ],
        dtype=torch.float32,
    )
    return predictions, targets, torch.ones_like(targets, dtype=torch.bool)


def test_soft_spearman_is_the_sole_objective() -> None:
    predictions, targets, mask = _rank_case()
    total, count = _soft_spearman_loss_sum(predictions, targets, mask)
    torch.testing.assert_close(total / count, soft_spearman_loss(predictions, targets, mask))
    assert objective_metadata() == {"name": "soft_spearman", "temperature": 0.5}


class TinyRanker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        patches: torch.Tensor,
        history_patch_mask: torch.Tensor,
        instrument_mask: torch.Tensor,
        slow_features: torch.Tensor,
        state_position: torch.Tensor,
    ) -> torch.Tensor:
        del history_patch_mask, instrument_mask, slow_features, state_position
        score = patches[:, :4, 0, 0] * self.weight
        return torch.stack((score, score.square()), dim=-1)


def test_compiled_soft_objective_has_finite_two_pass_sam_gradients() -> None:
    runtime = RuntimeSettings(
        effective_batch_size=2,
        loader_batch_size=2,
        microbatch_size=2,
        evaluation_batch_size=2,
        num_workers=0,
        compile_backend="eager",
    )
    model = TinyRanker()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    _, targets, mask = _rank_case()
    patches = torch.zeros(2, 4, 1, 1)
    patches[:, :, 0, 0] = torch.tensor(
        [[0.1, 0.2, 0.4, 0.3], [0.4, 0.1, 0.2, 0.3]]
    )
    batch = {
        "patches": patches,
        "history_patch_mask": torch.ones(2, 4, 1, dtype=torch.bool),
        "instrument_mask": torch.ones(2, 4, dtype=torch.bool),
        "slow_features": torch.zeros(2, 4, 1),
        "state_position": torch.ones(2, dtype=torch.long),
        "targets": targets,
        "label_mask": mask,
    }
    callbacks = 0

    def after_update() -> None:
        nonlocal callbacks
        callbacks += 1

    result = run_effective_batch_update(
        model,
        [batch],
        optimizer,
        None,
        runtime,
        training_objective=compile_training_objective(runtime),
        after_update=after_update,
    )
    assert result["backward_passes"] == 2
    assert torch.isfinite(result["gradient_norm"])
    assert callbacks == 1


def test_model_has_no_residual_attention_branch() -> None:
    model = SharedCausalTCN()
    assert not any("attention" in name for name, _ in model.named_parameters())
    assert model_metadata()["cross_equity_attention"] is False


def test_ema_and_weight_averages_are_exact() -> None:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = ModelEMA(model, 0.99)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)
    torch.testing.assert_close(ema.shadow["weight"], torch.full((1, 2), 1.02))
    averaged = average_state_dicts(
        ({"weight": torch.ones(2)}, {"weight": torch.full((2,), 3.0)})
    )
    torch.testing.assert_close(averaged["weight"], torch.full((2,), 2.0))


def test_early_stop_and_best_epoch_are_diagnostics_only() -> None:
    scores = [0.01, 0.02, 0.019, 0.018, 0.017] + [0.0] * (MAX_EPOCHS - 5)
    diagnostic = simulate_patience3(scores)
    assert diagnostic == {
        "selected_epoch": 2,
        "selected_score": 0.02,
        "stopped_epoch": 5,
        "selection_eligible": False,
    }
    assert retrospective_best_epoch(scores) == 2


def test_training_cli_exposes_only_current_controls() -> None:
    actions = {name for name, _ in parse_args([])._get_kwargs()}
    assert actions == {
        "seed",
        "selection_window",
        "selection_rule_file",
        "output_base",
    }


def test_repository_commit_is_independent_of_process_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = repository_commit()
    monkeypatch.chdir(tmp_path)
    assert repository_commit() == expected


def test_checkpoint_contains_raw_and_all_ema_states(tmp_path: Path) -> None:
    model = SharedCausalTCN()
    metadata = model_metadata()
    provenance = {
        "model": metadata,
        "feature_store_identity": {"path": str(tmp_path)},
        "repository_commit": "sha",
    }
    state = model.state_dict()
    payload = checkpoint_payload(
        model,
        {"ema_098": state, "ema_099": state, "ema_0995": state},
        seed=11,
        epoch=1,
        validation_scores={"raw": 0.01},
        feature_store=tmp_path,
        run_provenance=provenance,
    )
    assert payload["model"] == metadata
    assert set(payload["ema_state_dicts"]) == {"ema_098", "ema_099", "ema_0995"}
    assert "optimizer_state_dict" not in payload
