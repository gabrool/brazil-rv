from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from brazil_rv.modeling.contract import CONTEXT_COUNT, TCNArchitecture
from brazil_rv.modeling.engine import soft_spearman_loss
from brazil_rv.modeling.metrics import primary_validation_score
from brazil_rv.modeling.model import (
    BETA_BUCKET_SLOW_INDEX,
    PHASE_A_MODEL_VARIANTS,
    VOLATILITY_BUCKET_SLOW_INDEX,
    SharedCausalTCN,
    model_variant_metadata,
)
from brazil_rv.modeling.phase_a import (
    crossfit_patience_observations,
    parse_args,
    promote_after_stage_one,
)
from brazil_rv.preprocessing.contract import SLOW_CHANNELS


def _architecture() -> TCNArchitecture:
    return TCNArchitecture(
        patch_input_width=10,
        width=8,
        swiglu_hidden_width=4,
        residual_blocks=6,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16, 32),
        slow_width=32,
        fusion_states=CONTEXT_COUNT + 3,
        fusion_width=16,
        dropout=0.0,
        output_horizons=3,
    )


def _model_inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(5)
    instruments = 4 + CONTEXT_COUNT
    patches = torch.randn(2, instruments, 69, 10, generator=generator)
    history = torch.zeros(2, instruments, 69, dtype=torch.bool)
    history[:, :, :20] = True
    history[:, 4 + 7 :, :] = True
    instrument_mask = torch.ones(2, instruments, dtype=torch.bool)
    instrument_mask[:, 3] = False
    slow = torch.randn(2, instruments, 32, generator=generator)
    positions = torch.tensor([20, 20])
    return patches, history, instrument_mask, slow, positions


def test_phase_a_variants_start_as_the_exact_parent() -> None:
    architecture = _architecture()
    torch.manual_seed(17)
    parent = SharedCausalTCN(
        architecture=architecture, equity_count=4, variant="parent"
    ).eval()
    parent_state = parent.state_dict()
    inputs = _model_inputs()
    expected = parent(*inputs)

    for variant in PHASE_A_MODEL_VARIANTS:
        torch.manual_seed(17)
        candidate = SharedCausalTCN(
            architecture=architecture, equity_count=4, variant=variant
        ).eval()
        candidate_state = candidate.state_dict()
        for name, value in parent_state.items():
            torch.testing.assert_close(candidate_state[name], value, rtol=0, atol=0)
        adapters = [
            parameter
            for name, parameter in candidate.named_parameters()
            if "adapter" in name
        ]
        assert adapters
        assert all(torch.count_nonzero(parameter) == 0 for parameter in adapters)
        torch.testing.assert_close(candidate(*inputs), expected, rtol=0, atol=0)
        assert model_variant_metadata(variant)["zero_initialized_residual_adapter"]


def test_zero_started_adapters_receive_finite_rank_gradients() -> None:
    architecture = _architecture()
    inputs = _model_inputs()
    rank = torch.linspace(-1.0, 1.0, 4)[None, :, None]
    targets = rank.expand(2, -1, 3).clone()
    label_mask = inputs[2][:, :4, None].expand(-1, -1, 3)
    for variant in PHASE_A_MODEL_VARIANTS:
        torch.manual_seed(19)
        model = SharedCausalTCN(
            architecture=architecture, equity_count=4, variant=variant
        ).eval()
        loss = soft_spearman_loss(model(*inputs), targets, label_mask)
        loss.backward()
        adapter_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "adapter" in name
        ]
        assert adapter_gradients
        assert all(gradient is not None for gradient in adapter_gradients)
        assert all(torch.isfinite(gradient).all() for gradient in adapter_gradients)
        assert any(torch.count_nonzero(gradient) for gradient in adapter_gradients)


def test_cross_sectional_adapters_ignore_inactive_equities() -> None:
    architecture = _architecture()
    inputs = list(_model_inputs())
    for variant in (
        "cross_section_max_min",
        "learned_set_pool",
        "conditional_bucket_means",
    ):
        torch.manual_seed(23)
        model = SharedCausalTCN(
            architecture=architecture, equity_count=4, variant=variant
        ).eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "adapter" in name:
                    parameter.fill_(0.01)
        baseline = model(*inputs)
        mutated = [value.clone() for value in inputs]
        mutated[0][:, 3] = 1e4
        mutated[3][:, 3] = -1e4
        changed = model(*mutated)
        torch.testing.assert_close(changed[:, :3], baseline[:, :3], rtol=0, atol=0)


def test_temporal_adapters_ignore_masked_future_patches() -> None:
    architecture = _architecture()
    inputs = list(_model_inputs())
    for variant in ("temporal_stats", "multi_depth_stats"):
        torch.manual_seed(29)
        model = SharedCausalTCN(
            architecture=architecture, equity_count=4, variant=variant
        ).eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "adapter" in name:
                    parameter.fill_(0.01)
        baseline = model(*inputs)
        mutated = [value.clone() for value in inputs]
        mutated[0][:, : 4 + 7, 20:] = 1e4
        changed = model(*mutated)
        torch.testing.assert_close(changed, baseline, rtol=0, atol=0)


def test_conditional_pool_uses_active_causal_slow_fields() -> None:
    assert SLOW_CHANNELS[BETA_BUCKET_SLOW_INDEX] == "beta_to_WDO"
    assert (
        SLOW_CHANNELS[VOLATILITY_BUCKET_SLOW_INDEX] == "realized_vol_cross_section_rank"
    )
    assert BETA_BUCKET_SLOW_INDEX != SLOW_CHANNELS.index("beta_to_WIN")


def _write_patience_run(path: Path) -> None:
    path.mkdir()
    predictions_dir = path / "validation_predictions"
    predictions_dir.mkdir()
    dates = 102
    equities = 32
    targets = np.broadcast_to(
        np.linspace(-1.0, 1.0, equities, dtype=np.float32)[None, :, None],
        (dates, equities, 3),
    ).copy()
    odd_good = targets.copy()
    odd_good[1::2] *= -1
    even_good = -odd_good
    np.savez(
        path / "validation_reference.npz",
        targets=targets,
        raw_returns=targets,
        label_mask=np.ones_like(targets, dtype=bool),
        sample_id=np.arange(dates, dtype=np.int64),
        date_idx=np.arange(dates, dtype=np.int64),
        decision_idx=np.zeros(dates, dtype=np.int64),
    )
    for epoch in range(1, 21):
        values = odd_good if epoch == 1 else even_good
        np.savez(predictions_dir / f"epoch_{epoch:02d}.npz", raw=values)
    (path / "run_manifest.json").write_text(json.dumps({"seed": 29}), encoding="utf-8")


def test_patience_is_selected_on_one_parity_and_scored_on_the_other(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    _write_patience_run(run)
    observations, directions = crossfit_patience_observations(run)
    assert [row["selected_epoch"] for row in directions] == [1, 2]
    assert (
        primary_validation_score(
            observations.predictions,
            observations.targets,
            observations.label_mask,
            observations.date_idx,
        )
        == -1.0
    )


def _promotion_summary(patience: tuple[float, float], ema: tuple[float, float]):
    return {
        "readouts": {
            "patience3_raw": {
                "folds": {
                    "fold_a": {"candidate_minus_parent_primary_ic": patience[0]},
                    "fold_b": {"candidate_minus_parent_primary_ic": patience[1]},
                }
            },
            "final_ema_0995": {
                "folds": {
                    "fold_a": {"candidate_minus_parent_primary_ic": ema[0]},
                    "fold_b": {"candidate_minus_parent_primary_ic": ema[1]},
                }
            },
        }
    }


def test_stage_one_stops_only_concordant_large_losers() -> None:
    assert not promote_after_stage_one(
        _promotion_summary((-0.004, -0.003), (-0.002, -0.003))
    )
    assert promote_after_stage_one(
        _promotion_summary((-0.004, -0.0029), (-0.004, -0.004))
    )
    assert promote_after_stage_one(
        _promotion_summary((-0.005, -0.005), (-0.0019, -0.005))
    )


def test_phase_a_cli_exposes_only_parent_and_output_paths() -> None:
    actions = {
        name
        for name, _ in parse_args(
            ["--parent-campaign", "parent", "--output-dir", "output"]
        )._get_kwargs()
    }
    assert actions == {"parent_campaign", "output_dir"}
