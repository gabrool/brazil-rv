import copy

import pytest
import torch

from brazil_rv.v2.characteristic_model import CharacteristicConfig, CharacteristicModel
from brazil_rv.v2.model import encode_slow_history
from brazil_rv.v2.round7_training import TrainingObjective
from brazil_rv.v2.train import compile_forward, _unique_compiled_graphs


def inputs(dates=2, names=7):
    torch.manual_seed(37)
    history = torch.ones(dates, names, 60, dtype=torch.bool)
    history[:, 0] = False
    history[:, 1, :23] = False
    valid = history[..., None].expand(-1, -1, -1, 32).clone()
    valid[:, 2, 30:33] = False  # missing observations inside a real calendar suffix
    return dict(
        slow_features=torch.randn(dates, names, 60, 32),
        slow_feature_mask=valid,
        slow_history_mask=history,
        slow_feature_age_sessions=torch.zeros(dates, names, 60, 32),
        active_mask=torch.ones(dates, names, dtype=torch.bool),
    )


def test_restored_gru_last_state_exact_and_invalid_payload_ignored():
    model = CharacteristicModel(CharacteristicConfig()).eval()
    sample = inputs()
    args = {k: v for k, v in sample.items() if k != "active_mask"}
    kwargs = dict(
        config=model.temporal_config,
        input_projection=model.slow_input_projection,
        input_norm=model.slow_input_norm,
        encoder=model.slow_encoder,
    )
    old = encode_slow_history(**args, **kwargs)
    sequence = encode_slow_history(**args, **kwargs, return_sequence=True)
    torch.testing.assert_close(sequence[..., -1, :], old, atol=0, rtol=0)
    assert (sequence[~sample["slow_history_mask"]] == 0).all()


@pytest.mark.parametrize(
    "encoder,timing", [(e, t) for e in ("gru", "attention") for t in ("early", "late")]
)
def test_peer_masks_permutation_padding_and_no_cross_date_leakage(encoder, timing):
    torch.set_num_threads(2)
    model = CharacteristicModel(
        CharacteristicConfig(temporal_encoder=encoder, peer_timing=timing)
    ).eval()
    batch = inputs()
    with torch.no_grad():
        expected = model(**batch)
        order = torch.tensor([2, 5, 0, 6, 1, 3, 4])
        permuted = model(**{k: v[:, order] for k, v in batch.items()})
        torch.testing.assert_close(permuted, expected[:, order], atol=2e-6, rtol=2e-5)
        padded = {
            k: torch.cat((v, torch.zeros_like(v[:, :2])), dim=1)
            for k, v in batch.items()
        }
        padded["slow_features"][:, -2:] = float("nan")
        actual = model(**padded)
        torch.testing.assert_close(actual[:, :7], expected, atol=2e-6, rtol=2e-5)
        assert (actual[:, 7:] == 0).all()
        mutated = copy.deepcopy(batch)
        mutated["slow_features"][1] *= 100
        # Batch axis contains independent prediction dates; later decisions
        # cannot enter an earlier prediction through temporal or peer attention.
        torch.testing.assert_close(model(**mutated)[0], expected[0], atol=0, rtol=0)
        mutated = copy.deepcopy(batch)
        mutated["slow_features"][~mutated["slow_feature_mask"]] = float("nan")
        torch.testing.assert_close(model(**mutated), expected, atol=0, rtol=0)
    assert torch.isfinite(expected).all()
    model.train()
    result = model(**batch)
    result.square().mean().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


@pytest.mark.parametrize("encoder", ["gru", "attention"])
def test_early_late_have_identical_parameter_contract(encoder):
    states = []
    for timing in ("early", "late"):
        torch.manual_seed(11)
        states.append(
            CharacteristicModel(
                CharacteristicConfig(temporal_encoder=encoder, peer_timing=timing)
            ).state_dict()
        )
    assert states[0].keys() == states[1].keys()
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], atol=0, rtol=0)


@pytest.mark.parametrize(
    "encoder,timing", [(e, t) for e in ("gru", "attention") for t in ("early", "late")]
)
def test_pathway_fullgraph_across_distinct_date_batch_sizes(encoder, timing):
    torch._dynamo.reset()
    model = CharacteristicModel(
        CharacteristicConfig(temporal_encoder=encoder, peer_timing=timing)
    )
    objective = TrainingObjective(
        model,
        characteristic=True,
        head_indices=[2, 3, 4],
        loss_kind="soft_spearman",
        cuda=False,
    )
    compiled = compile_forward(objective, backend="eager", mode=None)
    before = _unique_compiled_graphs()
    for dates in (3, 4):
        sample = inputs(dates)
        sample.update(
            targets=torch.rand(dates, 7, 5),
            target_mask=torch.ones(dates, 7, 5, dtype=torch.bool),
        )
        compiled(sample).backward()
        model.zero_grad(set_to_none=True)
    assert _unique_compiled_graphs() - before == 1
