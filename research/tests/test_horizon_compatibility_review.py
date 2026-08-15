from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from brazil_rv.modeling.baselines import SharedCausalTCN
from brazil_rv.modeling.contract import (
    ABSOLUTE_PATCH_COUNT,
    BASELINE_TCN_SETTINGS,
    CONTEXT_COUNT,
    PATCH_INPUT_WIDTH,
    PEER_STATE_WIDTH,
    SLOW_FEATURE_COUNT,
    architecture_for_model,
    peer_feature_metadata,
)
from brazil_rv.modeling.engine import objective_metadata, sam_metadata
from brazil_rv.modeling.evaluate import load_current_neural_run
from brazil_rv.modeling.model import build_neural_model


EQUITIES = 4
INSTRUMENTS = EQUITIES + CONTEXT_COUNT


def _model(readout: str = "final") -> SharedCausalTCN:
    settings = replace(BASELINE_TCN_SETTINGS, readout=readout)
    architecture = architecture_for_model("tcn", settings)
    model = build_neural_model("tcn", architecture, "selected", equity_count=EQUITIES)
    assert isinstance(model, SharedCausalTCN)
    return model


def _inputs(batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(917)
    return (
        torch.randn(
            batch_size,
            INSTRUMENTS,
            ABSOLUTE_PATCH_COUNT,
            PATCH_INPUT_WIDTH,
            generator=generator,
        ),
        torch.ones(batch_size, INSTRUMENTS, ABSOLUTE_PATCH_COUNT, dtype=torch.bool),
        torch.ones(batch_size, INSTRUMENTS, dtype=torch.bool),
        torch.randn(
            batch_size,
            INSTRUMENTS,
            SLOW_FEATURE_COUNT,
            generator=generator,
        ),
        torch.full((batch_size,), 20, dtype=torch.long),
        torch.randn(
            batch_size,
            EQUITIES,
            PEER_STATE_WIDTH,
            generator=generator,
        ),
    )


def test_final_forward_matches_independent_pre_readout_reference_exactly() -> None:
    torch.manual_seed(29)
    model = _model().eval()
    inputs = _inputs()
    with torch.no_grad():
        states, taps = model._instrument_states(*inputs[:5])
        assert taps == ()
        equity_mask = inputs[2][:, :EQUITIES]
        context_mask = inputs[2][:, EQUITIES:]
        assert model.peer_adapter is not None
        equity = states[:, :EQUITIES] + model.peer_adapter(inputs[5])
        context = states[:, EQUITIES:]
        context_flat = (context * context_mask[..., None].to(context.dtype)).reshape(
            context.shape[0], CONTEXT_COUNT * model.architecture.width
        )
        weight = equity_mask[..., None].to(equity.dtype)
        count = weight.sum(dim=1).clamp_min(1.0)
        mean = (equity * weight).sum(dim=1) / count
        second = (equity.square() * weight).sum(dim=1) / count
        dispersion = torch.sqrt(torch.clamp(second - mean.square(), min=1e-6))
        shared = torch.cat((context_flat, mean, dispersion), dim=-1)
        shared = shared[:, None].expand(-1, EQUITIES, -1)
        fused = model.fusion_output(
            F.gelu(model.fusion_input(torch.cat((equity, shared), dim=-1)))
        )
        fused = F.gelu(fused)
        gate = torch.sigmoid(model.fusion_gate(torch.cat((equity, fused), dim=-1)))
        representation = model.fusion_norm(equity + gate * fused)
        expected = model.prediction_head(representation)
        expected *= equity_mask[..., None]
        actual = model(*inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "readout",
    ("final", "shared_multiscale", "horizon_multiscale", "final_score_mlp"),
)
def test_all_readouts_capture_as_static_full_graph(readout: str) -> None:
    torch.manual_seed(29)
    model = _model(readout).eval()
    compiled = torch.compile(
        model,
        backend="eager",
        fullgraph=True,
        dynamic=False,
    )
    output = compiled(*_inputs(batch_size=1))
    assert output.shape == (1, EQUITIES, 3)
    assert torch.isfinite(output).all()
    if readout == "horizon_multiscale":
        output.square().sum().backward()
        assert model.scale_logits is not None
        assert model.scale_logits.grad is not None
        assert torch.isfinite(model.scale_logits.grad).all()


def test_real_evaluation_path_loads_genuinely_old_final_checkpoint(
    tmp_path: Path,
) -> None:
    settings = BASELINE_TCN_SETTINGS
    architecture = architecture_for_model("tcn", settings)
    torch.manual_seed(29)
    original = build_neural_model("tcn", architecture, "selected").eval()
    old_settings = asdict(settings)
    old_settings.pop("readout")
    store = tmp_path / "store"
    store.mkdir()
    payload = {
        "model_name": "tcn",
        "architecture": asdict(architecture),
        "tcn_settings": old_settings,
        "peer_features": peer_feature_metadata("tcn", architecture, "selected"),
        "optimizer_variant": "sam_adamw",
        "objective": objective_metadata("soft_spearman", 0.50),
        "sam": sam_metadata("sam_adamw", 0.125),
        "seed": 29,
        "epoch": 2,
        "validation_score": 0.1,
        "feature_store": str(store),
        "global_context": "enabled",
        "model_state_dict": original.state_dict(),
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save(payload, run_dir / "best_checkpoint.pt")
    restored, checkpoint, restored_store = load_current_neural_run(run_dir)
    assert restored_store == store
    assert checkpoint["tcn_settings"]["readout"] == "final"
    assert checkpoint["training_horizon"] == "all"
    assert checkpoint["selection_horizon"] == "all"
    assert checkpoint["context_family_ablation"] == "none"

    instrument_count = original.instrument_count
    inputs = (
        torch.zeros(1, instrument_count, ABSOLUTE_PATCH_COUNT, PATCH_INPUT_WIDTH),
        torch.ones(1, instrument_count, ABSOLUTE_PATCH_COUNT, dtype=torch.bool),
        torch.ones(1, instrument_count, dtype=torch.bool),
        torch.zeros(1, instrument_count, SLOW_FEATURE_COUNT),
        torch.tensor([20]),
        torch.zeros(1, original.equity_count, PEER_STATE_WIDTH),
    )
    with torch.no_grad():
        expected = original(*inputs)
        actual = restored.eval()(*inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
