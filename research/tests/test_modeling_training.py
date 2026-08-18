from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from brazil_rv.modeling.contract import (
    ALLOWED_SEEDS,
    RECENCY_POLICIES,
    RuntimeSettings,
)
from brazil_rv.modeling.engine import (
    _soft_spearman_loss_sum,
    compile_training_objective,
    run_effective_batch_update,
    soft_spearman_loss,
)
from brazil_rv.modeling.model import SharedCausalTCN
from brazil_rv.modeling.run_core_campaign import (
    RunSpec,
    _completed_attempt,
    expand_campaign_specs,
    select_recency_parent,
)
from brazil_rv.modeling.train import parse_args


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


def test_uniform_loss_is_numerically_equivalent() -> None:
    predictions, targets, mask = _rank_case()
    ones = torch.ones(predictions.shape[0])
    expected = soft_spearman_loss(predictions, targets, mask)
    total, count = _soft_spearman_loss_sum(predictions, targets, mask, ones)
    torch.testing.assert_close(total / count, expected)


def test_weighted_loss_uses_weighted_valid_group_denominator() -> None:
    predictions, targets, mask = _rank_case()
    weights = torch.tensor([0.25, 2.0])
    total, count = _soft_spearman_loss_sum(predictions, targets, mask, weights)
    sample_losses = []
    for sample in range(2):
        value, groups = _soft_spearman_loss_sum(
            predictions[sample : sample + 1],
            targets[sample : sample + 1],
            mask[sample : sample + 1],
            torch.ones(1),
        )
        sample_losses.append(value / groups)
    expected = (
        weights[0] * sample_losses[0] + weights[1] * sample_losses[1]
    ) / weights.sum()
    torch.testing.assert_close(total / count, expected)


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


def test_compiled_weighted_objective_has_finite_two_pass_sam_gradients() -> None:
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
    predictions, targets, mask = _rank_case()
    del predictions
    patches = torch.zeros(2, 4, 1, 1)
    patches[:, :, 0, 0] = torch.tensor([[0.1, 0.2, 0.4, 0.3], [0.4, 0.1, 0.2, 0.3]])
    batch = {
        "patches": patches,
        "history_patch_mask": torch.ones(2, 4, 1, dtype=torch.bool),
        "instrument_mask": torch.ones(2, 4, dtype=torch.bool),
        "slow_features": torch.zeros(2, 4, 1),
        "state_position": torch.ones(2, dtype=torch.long),
        "targets": targets,
        "label_mask": mask,
        "training_weight": torch.tensor([0.5, 1.5]),
    }
    result = run_effective_batch_update(
        model,
        [batch],
        optimizer,
        None,
        runtime,
        training_objective=compile_training_objective(runtime),
    )
    assert result["backward_passes"] == 2
    assert torch.isfinite(result["gradient_norm"])


def test_attention_is_zero_initialized_parent_equivalent() -> None:
    model = SharedCausalTCN(cross_equity_attention=True, equity_count=6).eval()
    states = torch.randn(2, 6, 64)
    mask = torch.ones(2, 6, dtype=torch.bool)
    torch.testing.assert_close(model._attend_equities(states, mask), states)


def test_attention_is_permutation_equivariant_and_inactive_safe() -> None:
    model = SharedCausalTCN(cross_equity_attention=True, equity_count=6).eval()
    nn.init.normal_(model.equity_attention.out_proj.weight)
    states = torch.randn(2, 6, 64)
    mask = torch.tensor(
        [[True, True, False, True, False, True], [True, False, True, True, True, False]]
    )
    baseline = model._attend_equities(states, mask)
    permutation = torch.tensor([3, 0, 5, 1, 4, 2])
    permuted = model._attend_equities(states[:, permutation], mask[:, permutation])
    torch.testing.assert_close(permuted, baseline[:, permutation], atol=1e-5, rtol=1e-5)
    changed = states.clone()
    changed[~mask] = 1e6
    isolated = model._attend_equities(changed, mask)
    torch.testing.assert_close(isolated[mask], baseline[mask], atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(isolated[~mask]) == 0


def test_training_cli_exposes_only_campaign_switches() -> None:
    actions = {name for name, _ in parse_args([])._get_kwargs()}
    assert actions == {
        "seed",
        "recency_policy",
        "cross_equity_attention",
        "output_base",
    }


def test_campaign_expands_exactly_twenty_one_ordered_specs() -> None:
    specs = expand_campaign_specs()
    assert len(specs) == 21
    assert [spec.seed for spec in specs[:3]] == list(ALLOWED_SEEDS)
    assert {spec.recency_policy for spec in specs[6:18]} == set(RECENCY_POLICIES[1:])
    assert all(spec.recency_policy == "selected_parent" for spec in specs[18:])


def _completed_run(path: Path, seed: int, score: float, policy: str) -> Path:
    path.mkdir(parents=True)
    (path / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "seed": seed,
                "recency_policy": policy,
                "cross_equity_attention": False,
                "feature_store": str(path.parent.parent.parent / "store"),
                "repository_commit": "test",
                "split": {"test_accessed": False},
                "best_validation_score": score,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_recency_selection_requires_mean_and_two_matched_wins(
    tmp_path: Path,
) -> None:
    arms: dict[str, dict[int, Path]] = {}
    scores = {
        "uniform": [0.040, 0.041, 0.042],
        "exp_504": [0.041, 0.042, 0.041],
        "exp_252": [0.039, 0.045, 0.046],
        "exp_126": [0.043, 0.039, 0.041],
        "rolling_504": [0.038, 0.039, 0.050],
    }
    for policy, values in scores.items():
        arms[policy] = {
            seed: _completed_run(
                tmp_path / policy / f"seed_{seed}", seed, score, policy
            )
            for seed, score in zip(ALLOWED_SEEDS, values, strict=True)
        }
    assert select_recency_parent(arms) == "exp_252"


def test_resume_accepts_only_a_matching_completed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "store"
    store.mkdir()
    arm = tmp_path / "arm"
    attempt = arm / "attempt_01"
    attempt.mkdir(parents=True)
    spec = RunSpec("training", "tod_uniform", 11, "full_tod", "uniform", False)
    monkeypatch.setattr(
        "brazil_rv.modeling.run_core_campaign.repository_commit", lambda: "sha"
    )
    (attempt / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "seed": 11,
                "recency_policy": "uniform",
                "cross_equity_attention": False,
                "feature_store": str(store),
                "repository_commit": "sha",
                "split": {"test_accessed": False},
            }
        ),
        encoding="utf-8",
    )
    assert _completed_attempt(arm, spec, store) == attempt
