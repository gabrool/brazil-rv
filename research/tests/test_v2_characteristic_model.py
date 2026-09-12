from dataclasses import replace

import pytest
import torch

from brazil_rv.v2.characteristic_model import CharacteristicConfig, CharacteristicModel


def inputs():
    torch.manual_seed(11)
    values = torch.randn(2, 6, 60, 32)
    active = torch.tensor([[True, True, True, True, False, False]] * 2)
    side = torch.randn(2, 6, 3)
    return {
        "slow_features": values,
        "slow_feature_mask": torch.ones_like(values, dtype=torch.bool),
        "slow_history_mask": torch.ones(values.shape[:-1], dtype=torch.bool),
        "slow_feature_age_sessions": torch.zeros_like(values),
        "active_mask": active,
        "sidecars": {
            "fundamentals": (
                side,
                torch.ones_like(side, dtype=torch.bool),
                torch.zeros_like(side),
            )
        },
        "common_state": (
            torch.randn(2, 2),
            torch.ones(2, 2, dtype=torch.bool),
            torch.zeros(2, 2),
        ),
    }


@pytest.mark.parametrize(
    "context,members", [("pool", 1), ("attention", 1), ("pool", 8)]
)
def test_context_is_permutation_equivariant_and_padding_cannot_change_real_names(
    context, members
):
    torch.set_num_threads(1)
    batch = inputs()
    config = CharacteristicConfig(
        family_counts=(("fundamentals", 3),),
        common_field_count=2,
        context=context,
        members=members,
    )
    model = CharacteristicModel(config).eval()
    with torch.no_grad():
        expected = model(**batch)
        permutation = torch.tensor([3, 5, 1, 0, 4, 2])
        moved = {
            k: v[:, permutation] if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        moved["sidecars"] = {
            k: tuple(v[:, permutation] for v in fields)
            for k, fields in batch["sidecars"].items()
        }
        actual = model(**moved)
        torch.testing.assert_close(
            actual, expected[:, permutation], atol=2e-6, rtol=2e-5
        )
        batch["slow_features"][:, 4:] *= 1000
        batch["sidecars"]["fundamentals"][0][:, 4:] *= 1000
        torch.testing.assert_close(
            model(**batch)[:, :4], expected[:, :4], atol=2e-6, rtol=2e-5
        )
    assert expected.shape == (2, 6, members, 3)
    assert torch.count_nonzero(expected[:, 4:]) == 0


def test_invalid_families_and_initial_film_are_exactly_neutral():
    torch.set_num_threads(1)
    batch = inputs()
    config = CharacteristicConfig(
        family_counts=(("fundamentals", 3),), common_field_count=2
    )
    model = CharacteristicModel(config).eval()
    x, valid, age = batch["sidecars"]["fundamentals"]
    valid.zero_()
    with torch.no_grad():
        before = model(**batch)
        x.fill_(float("nan"))
        age.fill_(200.0)
        batch["common_state"][0].mul_(100.0)
        assert torch.equal(model(**batch), before)


def test_tabm_encodes_history_once_and_trains_all_members():
    torch.set_num_threads(1)
    batch = inputs()
    model = CharacteristicModel(
        CharacteristicConfig(
            family_counts=(("fundamentals", 3),), common_field_count=2, members=8
        )
    )
    calls = []
    handle = model.slow_encoder.register_forward_hook(lambda *_: calls.append(1))
    scores = model(**batch)
    handle.remove()
    assert len(calls) == 1
    assert scores[0, 0].std(dim=0).min() > 0
    scores.square().mean().backward()
    for block in model.trunk:
        assert torch.isfinite(block.up.r.grad).all()
        assert (block.up.r.grad.abs().sum(dim=1) > 0).all()


def test_no_temporal_arm_has_no_unused_gru_parameters_and_five_head_arm_is_explicit():
    config = CharacteristicConfig(temporal=False)
    model = CharacteristicModel(config)
    assert not any("slow_encoder" in n for n, _ in model.named_parameters())
    five = CharacteristicModel(replace(config, horizons=(1, 2, 3, 5, 10)))
    assert model.head.out_features == 3
    assert five.head.out_features == 5


def test_entirely_padded_attention_date_stays_finite():
    torch.set_num_threads(1)
    batch = inputs()
    batch["active_mask"].zero_()
    model = CharacteristicModel(
        CharacteristicConfig(
            family_counts=(("fundamentals", 3),),
            common_field_count=2,
            context="attention",
        )
    )
    scores = model(**batch)
    assert torch.isfinite(scores).all()
    assert torch.count_nonzero(scores) == 0
    scores.sum().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
