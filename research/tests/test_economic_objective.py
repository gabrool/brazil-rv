import json

import numpy as np
import torch

from brazil_rv.v2.artifacts import sha256_file
from brazil_rv.v2.characteristic_model import CharacteristicConfig, CharacteristicModel
from brazil_rv.v2.config import ModelConfig
from brazil_rv.v2.economic_objective import (
    EconomicCollator,
    attach_economic_head,
    economic_loss,
)
from brazil_rv.v2.model import DailyMultiHorizonModel


def test_auxiliary_preserves_predictions_rng_and_reaches_both_encoders():
    torch.set_num_threads(1)
    for model in (
        DailyMultiHorizonModel(
            ModelConfig(
                slow_feature_count=4,
                current_feature_count=0,
                disable_fast_stream=True,
                dropout=0,
            )
        ),
        CharacteristicModel(
            CharacteristicConfig(
                slow_feature_count=4,
                width=32,
                inner_width=32,
                blocks=1,
                dropout=0,
                temporal_encoder="attention",
                peer_timing="early",
            )
        ),
    ):
        values = torch.randn(2, 4, 60, 4)
        valid = torch.ones_like(values, dtype=torch.bool)
        history = torch.ones((2, 4, 60), dtype=torch.bool)
        active = torch.ones((2, 4), dtype=torch.bool)
        arguments = (values, valid, history, active)
        kwargs = {"slow_feature_age_sessions": torch.zeros_like(values)}
        before = model(*arguments, **kwargs)
        rng = torch.get_rng_state().clone()
        attach_economic_head(model)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        after, hidden = model(*arguments, **kwargs, return_hidden=True)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        if hidden.ndim == 3:
            hidden = hidden.unsqueeze(2)
        target = torch.randn(2, 4)
        prediction = model.economic_head(hidden).squeeze(-1)
        loss = economic_loss(prediction, target, active)
        loss.backward()
        assert model.economic_head.weight.grad.norm() > 0
        with torch.no_grad():
            model.economic_head.weight.add_(
                model.economic_head.weight.grad, alpha=-0.01
            )
        model.zero_grad(set_to_none=True)
        _, hidden = model(*arguments, **kwargs, return_hidden=True)
        if hidden.ndim == 3:
            hidden = hidden.unsqueeze(2)
        economic_loss(
            model.economic_head(hidden).squeeze(-1), target, active
        ).backward()
        assert model.slow_input_projection.weight.grad.norm() > 0


def test_economic_scaling_and_labels_are_fit_and_endpoint_contained(tmp_path):
    source = tmp_path / "targets.npz"
    values = np.arange(48, dtype=np.float32).reshape(12, 4) / 100
    valid = np.ones_like(values, bool)

    def save():
        np.savez(
            source,
            indices=np.arange(12),
            values=values,
            valid=valid,
            isins=np.asarray(["a", "b", "c", "d"]),
        )
        source.with_suffix(".json").write_text(
            json.dumps({"sha256": sha256_file(source)})
        )

    save()
    batch = {
        "date_index": torch.tensor([0, 2, 7]),
        "name_index": torch.tensor([[2, 0, -1]] * 3),
    }
    first = EconomicCollator(
        lambda _: dict(batch), source, np.arange(5), np.arange(7), ("a", "b", "c", "d")
    )
    a = first([])
    assert a["economic_mask"].tolist() == [
        [True, True, False],
        [False, False, False],
        [False, False, False],
    ]
    values[5:] = 1e6
    save()
    second = EconomicCollator(
        lambda _: dict(batch), source, np.arange(5), np.arange(7), ("a", "b", "c", "d")
    )
    assert first.contract["scale"] == second.contract["scale"]
    torch.testing.assert_close(
        a["economic_target"], second([])["economic_target"], rtol=0, atol=0
    )
    prediction = torch.randn(3, 3, 1, requires_grad=True)
    economic_loss(prediction, a["economic_target"], a["economic_mask"]).backward()
    assert prediction.grad[1:].count_nonzero() == 0
