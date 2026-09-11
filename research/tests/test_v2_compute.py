from copy import deepcopy
from functools import partial

import numpy as np
import pytest
import torch

from brazil_rv.v2.config import ModelConfig
from brazil_rv.v2.data import V2DailyDataset, collate_v2_daily, restore_name_axis
from brazil_rv.v2.losses import multi_horizon_loss, multi_horizon_loss_normalizers
from brazil_rv.v2.model import DailyMultiHorizonModel
from brazil_rv.v2.train import DateBatchSampler, _model_forward, fit_date_weights
from brazil_rv.modeling.engine import _soft_spearman_loss_sum


def test_compaction_preserves_eligible_history_predictions_and_gradients():
    rng = np.random.default_rng(11)
    samples = []
    for indices in ([1, 4, 8, 10, 18], [0, 3, 7, 15]):
        active = np.zeros(23, dtype=bool)
        active[indices] = True
        slow = rng.normal(size=(23, 60, 3)).astype(np.float32)
        mask = np.repeat(active[:, None], 5, axis=1)
        mask[indices[0]] = False  # Eligible but unlabeled: must remain in pooling.
        samples.append(
            dict(
                slow_features=slow,
                slow_feature_mask=np.ones_like(slow, dtype=bool),
                slow_history_mask=np.ones((23, 60), dtype=bool),
                slow_feature_age_sessions=np.zeros_like(slow),
                active_mask=active,
                targets=rng.normal(size=(23, 5)).astype(np.float32),
                target_mask=mask,
                sidecar_events_values=rng.normal(size=(23, 2)).astype(np.float32),
                sidecar_events_valid=np.ones((23, 2), dtype=bool),
                sidecar_events_age_sessions=np.zeros((23, 2), dtype=np.float32),
            )
        )
    dense = collate_v2_daily(samples)
    packed = collate_v2_daily(samples, fixed_name_count=16)
    for date, sample in enumerate(samples):
        names = np.flatnonzero(sample["active_mask"])
        np.testing.assert_array_equal(
            packed["slow_features"][date, : len(names)], sample["slow_features"][names]
        )
        np.testing.assert_array_equal(packed["name_index"][date, : len(names)], names)
    torch.manual_seed(11)
    model = DailyMultiHorizonModel(
        ModelConfig(
            slow_feature_count=3,
            current_feature_count=0,
            disable_fast_stream=True,
            sidecar_feature_counts=(("events", 2),),
            dropout=0,
            compile_forward=False,
        )
    ).eval()
    other = deepcopy(model)
    full = _model_forward(model, dense)[..., :5]
    compact = _model_forward(other, packed)[..., :5]
    restored = restore_name_axis(
        compact.detach().numpy(), packed["name_index"].numpy(), 23
    )
    np.testing.assert_allclose(full.detach().numpy(), restored, atol=2e-6, rtol=2e-5)
    losses = [
        multi_horizon_loss(v, b["targets"], b["target_mask"])
        for v, b in ((full, dense), (compact, packed))
    ]
    torch.testing.assert_close(*losses, atol=2e-6, rtol=2e-5)
    for loss in losses:
        loss.backward()
    for left, right in zip(model.parameters(), other.parameters(), strict=True):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=3e-6, rtol=5e-4)


def test_compaction_never_truncates_a_large_historical_universe():
    sample = {"active_mask": np.arange(933) < 243}
    packed = collate_v2_daily([sample], fixed_name_count=256)
    assert packed["active_mask"].sum() == 243
    assert packed["name_index"][0, 242] == 242
    with pytest.raises(ValueError, match="truncation is forbidden"):
        collate_v2_daily([sample], fixed_name_count=192)


def test_unique_date_epochs_keep_remainders_gaps_and_shuffle_deterministically():
    dates = np.r_[np.arange(100, 400), np.arange(450, 467)]
    sampler = DateBatchSampler(dates, seed=11)
    first = list(sampler)
    assert first == list(sampler)
    assert sorted(x for b in first for x in b) == list(range(len(dates)))
    assert all(2 <= len(b) <= 16 for b in first)
    sampler.set_epoch(1)
    assert first != list(sampler)
    assert sorted(x for b in sampler for x in b) == list(range(len(dates)))


def test_unique_date_decay_matches_full_weighted_objective_and_microbatch_gradient():
    torch.manual_seed(11)
    dates = np.arange(24) * 100
    mask = torch.ones((24, 7, 5), dtype=torch.bool)
    mask[:10, :, 2] = False
    weights = fit_date_weights(dates, mask.numpy(), 756)
    valid = mask.sum(1) >= 2
    torch.testing.assert_close((weights * valid).sum(0), torch.full((5,), 24.0))
    scores = torch.randn(24, 7, 5, requires_grad=True)
    targets = torch.randn_like(scores)
    counts = multi_horizon_loss_normalizers(mask)
    counts["per_horizon"].fill_(24)
    full = multi_horizon_loss(
        scores, targets, mask, date_weights=weights, normalization_counts=counts
    )
    split = sum(
        multi_horizon_loss(
            scores[i : i + 8],
            targets[i : i + 8],
            mask[i : i + 8],
            date_weights=weights[i : i + 8],
            normalization_counts=counts,
        )
        for i in range(0, 24, 8)
    )
    torch.testing.assert_close(full, split)
    torch.testing.assert_close(
        torch.autograd.grad(full, scores, retain_graph=True)[0],
        torch.autograd.grad(split, scores)[0],
    )


def test_vectorized_horizons_preserve_separate_head_objective_and_gradients():
    torch.manual_seed(47)
    scores = torch.randn(6, 13, 6, requires_grad=True)
    targets = torch.randn_like(scores)
    mask = torch.rand_like(scores) > 0.3
    mask[:3, :, 1] = False
    mask[0, :, 4] = False
    individual = []
    for h in range(6):
        total, count = _soft_spearman_loss_sum(
            scores[..., h : h + 1], targets[..., h : h + 1], mask[..., h : h + 1], 0.1
        )
        individual.append(total / count.clamp_min(1))
    expected = torch.stack(individual[:5]).mean() + 0.2 * individual[5]
    actual = multi_horizon_loss(
        scores, targets, mask, temperature=0.1, to_close_weight=0.2
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, scores, retain_graph=True)[0],
        torch.autograd.grad(expected, scores)[0],
    )


def test_early_history_packing_and_scoring_keep_canonical_identities(tmp_path):
    from torch.utils.data import DataLoader
    from brazil_rv.v2.score import score_checkpoint_artifact
    from test_v2_score import _scoring_fixture

    dense, config, checkpoint, _ = _scoring_fixture(tmp_path)
    packed = V2DailyDataset(
        dense.store.root,
        dense.date_indices,
        stage=dense.stage,
        lookback=dense.lookback,
        purpose="evaluation",
        compact_names=True,
    )
    dense_rows = [dense[i] for i in range(len(dense))]
    packed_rows = [packed[i] for i in range(len(packed))]
    left = collate_v2_daily(dense_rows, fixed_name_count=16)
    right = collate_v2_daily(packed_rows, fixed_name_count=16)
    for key in left:
        if isinstance(left[key], torch.Tensor):
            assert torch.equal(left[key], right[key]), key
        else:
            assert left[key] == right[key]
    for name, dataset, collate in (
        ("dense", dense, collate_v2_daily),
        ("packed", packed, partial(collate_v2_daily, fixed_name_count=16)),
    ):
        score_checkpoint_artifact(
            checkpoint=checkpoint,
            model_config=config,
            loader=DataLoader(dataset, batch_size=2, collate_fn=collate),
            output_dir=tmp_path / name,
            device=torch.device("cpu"),
        )
    np.testing.assert_array_equal(
        np.load(tmp_path / "dense/score_mask.npy"),
        np.load(tmp_path / "packed/score_mask.npy"),
    )
    np.testing.assert_allclose(
        np.load(tmp_path / "dense/scores.npy"),
        np.load(tmp_path / "packed/scores.npy"),
        atol=2e-6,
        rtol=2e-5,
    )
