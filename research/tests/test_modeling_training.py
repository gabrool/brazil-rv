from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from brazil_rv.modeling.contract import RuntimeSettings
from brazil_rv.modeling.engine import (
    _gap_weighted_pairwise_loss_sum,
    _hybrid_loss_sum,
    _soft_spearman_loss_sum,
    checkpoint_payload,
    compile_training_objective,
    objective_metadata,
    run_effective_batch_update,
    soft_spearman_loss,
)
from brazil_rv.modeling.model import SharedCausalTCN
from brazil_rv.modeling.provenance import model_metadata
from brazil_rv.modeling.run_loss_attention_campaign import (
    RunSpec,
    _matching_completed_attempt,
    expand_campaign_specs,
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


def test_uniform_soft_loss_is_numerically_equivalent() -> None:
    predictions, targets, mask = _rank_case()
    ones = torch.ones(predictions.shape[0])
    expected = soft_spearman_loss(predictions, targets, mask)
    total, count = _soft_spearman_loss_sum(predictions, targets, mask, ones)
    torch.testing.assert_close(total / count, expected)


def test_weighted_soft_loss_uses_weighted_valid_group_denominator() -> None:
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


def test_gap_loss_prioritizes_large_continuous_return_gaps() -> None:
    targets = torch.tensor([[[-0.75], [0.0], [0.75]]])
    continuous = torch.tensor([[[-0.05], [0.0], [1.0]]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    near_swap = torch.tensor([[[0.1], [0.0], [1.0]]])
    far_errors = torch.tensor([[[1.0], [0.1], [0.0]]])
    weights = torch.ones(1)
    near = _gap_weighted_pairwise_loss_sum(
        near_swap, targets, continuous, mask, weights
    )
    far = _gap_weighted_pairwise_loss_sum(
        far_errors, targets, continuous, mask, weights
    )
    assert far > near


def test_hybrid_loss_retains_soft_spearman_component() -> None:
    predictions, targets, mask = _rank_case()
    continuous = targets * 0.5
    weights = torch.ones(predictions.shape[0])
    hybrid, count = _hybrid_loss_sum(predictions, targets, continuous, mask, weights)
    soft, soft_count = _soft_spearman_loss_sum(predictions, targets, mask, weights)
    assert count == soft_count
    assert hybrid > soft


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


def test_compiled_hybrid_objective_has_finite_two_pass_sam_gradients() -> None:
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
    patches[:, :, 0, 0] = torch.tensor([[0.1, 0.2, 0.4, 0.3], [0.4, 0.1, 0.2, 0.3]])
    batch = {
        "patches": patches,
        "history_patch_mask": torch.ones(2, 4, 1, dtype=torch.bool),
        "instrument_mask": torch.ones(2, 4, dtype=torch.bool),
        "slow_features": torch.zeros(2, 4, 1),
        "state_position": torch.ones(2, dtype=torch.long),
        "targets": targets,
        "continuous_targets": targets * 0.5,
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


def test_attention_relationship_states_remove_active_common_state() -> None:
    states = torch.randn(2, 6, 64)
    mask = torch.tensor(
        [[True, True, False, True, False, True], [True, False, True, True, True, False]]
    )
    specific = SharedCausalTCN._specific_attention_states(states, mask)
    weight = mask[..., None].to(states.dtype)
    torch.testing.assert_close(
        (specific * weight).sum(1), torch.zeros(2, 64), atol=1e-6, rtol=0.0
    )
    assert torch.count_nonzero(specific[~mask]) == 0


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


def test_training_cli_exposes_only_current_switches() -> None:
    actions = {
        name for name, _ in parse_args(["--target-scale-dir", "scale"])._get_kwargs()
    }
    assert actions == {
        "seed",
        "recency_policy",
        "cross_equity_attention",
        "target_scale_dir",
        "output_base",
    }


def test_campaign_expands_exactly_two_ordered_specs() -> None:
    specs = expand_campaign_specs()
    assert [spec.arm for spec in specs] == [
        "hybrid_base",
        "hybrid_residual_attention",
    ]
    assert [spec.cross_equity_attention for spec in specs] == [False, True]


def test_campaign_resume_requires_an_exact_completed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "brazil_rv.modeling.run_loss_attention_campaign.repository_commit",
        lambda: "sha",
    )
    store = tmp_path / "store"
    store.mkdir()
    arm = tmp_path / "arm"
    attempt = arm / "attempt_01"
    attempt.mkdir(parents=True)
    scale_identity = {"target_scale_sha256": "scale"}
    manifest_path = attempt / "run_manifest.json"
    manifest = {
        "status": "completed",
        "repository_commit": "sha",
        "seed": 11,
        "recency_policy": "uniform",
        "cross_equity_attention": False,
        "objective": objective_metadata(),
        "target_scale_identity": scale_identity,
        "feature_store": str(store),
        "split": {"test_accessed": False},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    spec = RunSpec("hybrid_base", False)
    assert _matching_completed_attempt(arm, spec, store, scale_identity) == attempt

    manifest["split"]["test_accessed"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _matching_completed_attempt(arm, spec, store, scale_identity) is None


def test_checkpoint_model_metadata_is_json_canonical(tmp_path: Path) -> None:
    model = SharedCausalTCN()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    metadata = model_metadata(False)
    assert isinstance(metadata["architecture"]["dilations"], list)
    scale_identity = {"path": str(tmp_path / "target_scale")}
    provenance = {
        "model": metadata,
        "feature_store_identity": {"path": str(tmp_path)},
        "target_scale_identity": scale_identity,
        "repository_commit": "sha",
    }
    payload = checkpoint_payload(
        model,
        optimizer,
        scheduler,
        cross_equity_attention=False,
        recency_policy="uniform",
        seed=11,
        epoch=1,
        validation_score=0.01,
        feature_store=tmp_path,
        target_scale_dir=tmp_path / "target_scale",
        target_scale_identity=scale_identity,
        run_provenance=provenance,
    )
    assert payload["model"] == provenance["model"]
