import copy

import numpy as np
import pytest
import torch
from torch import nn

from brazil_rv.v2.round7 import calibrate_budget
from brazil_rv.v2.round7_training import (
    TailAverage,
    daily_primary_ic,
    learning_rate_fraction,
    member_loss,
    optimizer_step,
    recipe_optimizer,
)
from brazil_rv.v2.train import _common_primary_selection_score, sam_accumulated_step


@pytest.mark.parametrize("kind", ["soft_spearman", "pearson_on_ranks"])
def test_member_loss_is_mean_of_separate_member_rank_losses(kind):
    torch.manual_seed(12)
    score = torch.randn(3, 23, 8, 3, requires_grad=True)
    target = torch.rand(3, 23, 3)
    mask = torch.rand(3, 23, 3) > 0.15
    actual = member_loss(score, target, mask, kind=kind)
    expected = torch.stack(
        [member_loss(score[:, :, k : k + 1], target, mask, kind=kind) for k in range(8)]
    ).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(score.grad).all()
    assert (score.grad.abs().sum((0, 1, 3)) > 0).all()


def test_foreach_sam_matches_reference_and_exactly_restores_on_failure():
    torch.manual_seed(9)
    first = nn.Sequential(nn.Linear(3, 5), nn.Dropout(0.2), nn.Linear(5, 1))
    second = copy.deepcopy(first)
    x, y = torch.randn(8, 3), torch.randn(8, 1)
    a, b = recipe_optimizer(first, cuda=False), recipe_optimizer(second, cuda=False)
    torch.manual_seed(19)
    optimizer_step(first, a, lambda: (first(x) - y).square().mean(), 0.05)
    torch.manual_seed(19)
    sam_accumulated_step(
        second,
        b,
        [lambda: (second(x) - y).square().mean()],
        rho=0.05,
        gradient_clip=1.0,
    )
    for p, q in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(p, q)
    before = copy.deepcopy(first.state_dict())
    calls = 0

    def fails():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("failed second pass")
        return first(x).square().mean()

    with pytest.raises(RuntimeError, match="failed second"):
        optimizer_step(first, a, fails, 0.05)
    for name, parameter in first.state_dict().items():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_tail_average_has_no_initial_weight_and_decay_excludes_norm_bias():
    m = nn.Sequential(nn.Linear(1, 1), nn.LayerNorm(1))
    optimizer = recipe_optimizer(m, cuda=False)
    assert optimizer.param_groups[0]["params"] == [m[0].weight]
    assert len(optimizer.param_groups[1]["params"]) == 3
    tail = TailAverage()
    with torch.no_grad():
        m[0].weight.fill_(2.0)
    tail.update(m)
    with torch.no_grad():
        m[0].weight.fill_(4.0)
    tail.update(m)
    assert tail.state["0.weight"].item() == 3.0
    assert tail.count == 2


def test_budget_and_learning_rate_endpoints_are_fixed_without_evaluation_scores():
    assert calibrate_budget(np.full((12, 60), 0.02)) == 20
    assert calibrate_budget(np.broadcast_to(np.linspace(0.0, 0.06, 60), (12, 60))) == 60
    assert learning_rate_fraction(0, 1000) == 0.02
    assert learning_rate_fraction(49, 1000) == 1.0
    assert learning_rate_fraction(999, 1000) == 0.05


def test_daily_selection_readout_matches_standing_common_population():
    rng = np.random.default_rng(4)
    scores, targets = rng.normal(size=(2, 32, 3)), rng.normal(size=(2, 32, 3))
    mask = rng.random((2, 32, 3)) > 0.05
    active = np.ones((2, 32), bool)
    actual = daily_primary_ic(scores, targets, mask, active).mean()
    expected = _common_primary_selection_score(
        scores, targets, mask, active, horizons=(3, 5, 10)
    )
    assert actual == pytest.approx(expected)


def test_three_head_export_never_fabricates_short_horizon_predictions():
    from brazil_rv.v2.round7_score import canonical_head_panel

    values = np.asarray([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], np.float32)
    active = np.asarray([[True, False]])
    scores, mask = canonical_head_panel(values, active, (3, 5, 10))
    assert mask[0, 0].tolist() == [False, False, True, True, True]
    np.testing.assert_array_equal(scores[0, 0], [0.0, 0.0, 1.0, 2.0, 3.0])
    assert not scores[0, 1].any() and not mask[0, 1].any()


def test_fixed_fit_resume_and_tail_scoring_are_identical(tmp_path, monkeypatch):
    import json
    from dataclasses import asdict
    from test_v2_training import _tracked_pretrain_loaders
    from brazil_rv.v2 import round7_training as training
    from brazil_rv.v2 import round7_score as scoring
    from brazil_rv.v2.artifacts import sha256_file
    from brazil_rv.v2.characteristic_model import CharacteristicModel
    from brazil_rv.v2.round7 import CELLS, configuration
    from v2_store_fixtures import write_fixture_store

    factory, selection_loader = _tracked_pretrain_loaders(tmp_path)
    root = factory().dataset.store.root
    factory().dataset.store.close()
    selection_loader.dataset.store.close()
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["metadata"]["round7_repair"] = {"synthetic_fixture": True}
    root = write_fixture_store(
        tmp_path / "round7_fixture",
        dates=np.load(root / "date_index.npy") + np.timedelta64(2922, "D"),
        isins=np.load(root / "isin_index.npy").tolist(),
        arrays={
            name: np.load(root / record["path"])
            for name, record in manifest["arrays"].items()
        },
        feature_names=manifest["feature_names"],
        metadata=manifest["metadata"],
    )
    manifest_path = root / "manifest.json"
    fit, selection, evaluation = (
        np.arange(100, 104),
        np.arange(120, 145),
        np.arange(160, 185),
    )

    def indices(*args):
        return fit, selection, evaluation, np.arange(100, 115)

    monkeypatch.setattr(training, "_cli_stage_indices", indices)
    monkeypatch.setattr(scoring, "_cli_stage_indices", indices)
    monkeypatch.setattr(training, "_git_identity", lambda: {"commit": "f" * 40})
    cell = next(c for c in CELLS if c["cell"] == "B3")
    config = configuration(cell, manifest["feature_names"])
    torch.manual_seed(22)
    parent = tmp_path / "parent.pt"
    torch.save(
        {
            "schema": training.CHECKPOINT_SCHEMA,
            "stage": "P",
            "seed": 11,
            "contract": {
                "pretrain_key": "c1_slow",
                "config": asdict(config),
                "store_manifest_sha256": sha256_file(manifest_path),
            },
            "model_state_dict": CharacteristicModel(config).state_dict(),
        },
        parent,
    )
    options = dict(
        cell_name="B3",
        stage="F",
        fold="F1",
        seed=11,
        epochs=8,
        parent=parent,
        parent_sha256=sha256_file(parent),
        device=torch.device("cpu"),
        compiled=False,
        export_scores=True,
    )
    full = tmp_path / "full"
    training.train(root, full, **options)
    original = training.write_json_atomic

    def interrupt(path, payload):
        if path.name == "history.json" and len(payload) == 6:
            raise RuntimeError("simulated interruption after durable epoch")
        return original(path, payload)

    resumed = tmp_path / "resumed"
    monkeypatch.setattr(training, "write_json_atomic", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        training.train(root, resumed, **options)
    monkeypatch.setattr(training, "write_json_atomic", original)
    training.train(root, resumed, **options)
    a = torch.load(full / "tail_average.pt", weights_only=True)
    b = torch.load(resumed / "tail_average.pt", weights_only=True)
    assert a["tail_count"] == b["tail_count"] == 2
    for key, value in a["model_state_dict"].items():
        torch.testing.assert_close(value, b["model_state_dict"][key], atol=0, rtol=0)
    for filename in (
        "scores.npy",
        "score_mask.npy",
        "date_index.npy",
        "isin_index.npy",
    ):
        np.testing.assert_array_equal(
            np.load(full / "scores" / filename), np.load(resumed / "scores" / filename)
        )
    assert training.train(root, resumed, **options)["status"] == "completed"


def test_corrector_purge_and_sequential_batches_keep_all_permitted_dates():
    from brazil_rv.v2.round7_corrector import mature_before
    from brazil_rv.v2.round7_training import sequential_batches

    np.testing.assert_array_equal(
        mature_before([88, 89, 90, 91], 100), [True, True, False, False]
    )
    for count in (17, 33, 49, 55, 65, 113):
        batches = sequential_batches(count)
        assert min(map(len, batches)) >= 2
        np.testing.assert_array_equal(np.concatenate(batches), np.arange(count))


@pytest.mark.parametrize(
    "context,members", [("pool", 1), ("attention", 1), ("pool", 8)]
)
def test_complete_objective_compiles_once_across_date_batch_sizes(context, members):
    from brazil_rv.v2.characteristic_model import (
        CharacteristicConfig,
        CharacteristicModel,
    )
    from brazil_rv.v2.round7_training import TrainingObjective
    from brazil_rv.v2.train import compile_forward, _unique_compiled_graphs

    torch._dynamo.reset()
    model = CharacteristicModel(CharacteristicConfig(context=context, members=members))
    objective = TrainingObjective(
        model,
        characteristic=True,
        head_indices=[2, 3, 4],
        loss_kind="soft_spearman",
        cuda=False,
    )
    compiled = compile_forward(objective, backend="eager", mode=None)
    before = _unique_compiled_graphs()
    for dates in (4, 5):
        batch = {
            "slow_features": torch.randn(dates, 24, 60, 32),
            "slow_feature_mask": torch.ones(dates, 24, 60, 32, dtype=torch.bool),
            "slow_history_mask": torch.ones(dates, 24, 60, dtype=torch.bool),
            "slow_feature_age_sessions": torch.zeros(dates, 24, 60, 32),
            "active_mask": torch.ones(dates, 24, dtype=torch.bool),
            "targets": torch.rand(dates, 24, 5),
            "target_mask": torch.ones(dates, 24, 5, dtype=torch.bool),
        }
        loss = compiled(batch)
        loss.backward()
        assert torch.isfinite(loss)
        model.zero_grad(set_to_none=True)
    assert _unique_compiled_graphs() - before == 1
