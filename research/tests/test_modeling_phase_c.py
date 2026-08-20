from __future__ import annotations

import torch

from brazil_rv.modeling.contract import CONTEXT_COUNT, TCNArchitecture
from brazil_rv.modeling.engine import soft_spearman_loss
from brazil_rv.modeling.model import (
    CAPACITY_96_VARIANT,
    COMPRESSED_GLOBAL_RISK_VARIANT,
    COMPETITIVE_FEATURE_GATE_VARIANT,
    DI_TILT_EXPOSURE_VARIANT,
    FACTOR_MIXER_K4_VARIANT,
    FACTOR_MIXER_K8_VARIANT,
    MODEL_VARIANTS,
    PARENT_MODEL_VARIANT,
    SET_POOL_FACTOR_MIXER_VARIANT,
    SharedCausalTCN,
)
from brazil_rv.modeling.phase_c import c2_extensions_allowed, parse_args

ZERO_START_VARIANTS = tuple(
    variant
    for variant in MODEL_VARIANTS
    if variant not in (PARENT_MODEL_VARIANT, CAPACITY_96_VARIANT)
)


def _architecture() -> TCNArchitecture:
    return TCNArchitecture(
        patch_input_width=10,
        width=8,
        swiglu_hidden_width=4,
        residual_blocks=2,
        kernel_size=3,
        dilations=(1, 2),
        slow_width=6,
        fusion_states=CONTEXT_COUNT + 3,
        fusion_width=16,
        dropout=0.0,
        output_horizons=3,
    )


def _inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(5)
    batch, equities, patches = 2, 4, 69
    instruments = equities + CONTEXT_COUNT
    instrument_mask = torch.ones(batch, instruments, dtype=torch.bool)
    instrument_mask[:, 3] = False
    return {
        "patches": torch.randn(batch, instruments, patches, 10, generator=generator),
        "history_patch_mask": torch.ones(batch, instruments, patches, dtype=torch.bool),
        "instrument_mask": instrument_mask,
        "slow_features": torch.randn(batch, instruments, 6, generator=generator),
        "state_position": torch.full((batch,), 20, dtype=torch.long),
        "market_state": torch.randn(batch, 4, generator=generator),
        "tilt_exposure": torch.randn(batch, equities, generator=generator),
    }


def test_phase_c_zero_start_variants_are_exact_parent() -> None:
    architecture = _architecture()
    inputs = _inputs()
    torch.manual_seed(17)
    parent = SharedCausalTCN(
        architecture=architecture, equity_count=4, variant=PARENT_MODEL_VARIANT
    ).eval()
    expected = parent(**inputs)
    parent_state = parent.state_dict()

    for variant in ZERO_START_VARIANTS:
        torch.manual_seed(17)
        candidate = SharedCausalTCN(
            architecture=architecture, equity_count=4, variant=variant
        ).eval()
        candidate_state = candidate.state_dict()
        for name, value in parent_state.items():
            torch.testing.assert_close(candidate_state[name], value, rtol=0, atol=0)
        torch.testing.assert_close(candidate(**inputs), expected, rtol=0, atol=0)


def _upstream_parameters(
    model: SharedCausalTCN, variant: str
) -> tuple[torch.nn.Parameter, ...]:
    if variant == COMPRESSED_GLOBAL_RISK_VARIANT:
        return tuple(model.market_state_encoder.parameters())
    if variant in (FACTOR_MIXER_K4_VARIANT, FACTOR_MIXER_K8_VARIANT):
        return (model.factor_queries, *model.factor_loadings.parameters())
    if variant == SET_POOL_FACTOR_MIXER_VARIANT:
        return (
            model.factor_queries,
            *model.factor_loadings.parameters(),
            *model.set_phi.parameters(),
        )
    if variant == COMPETITIVE_FEATURE_GATE_VARIANT:
        return tuple(model.feature_gate_encoder.parameters())
    if variant == DI_TILT_EXPOSURE_VARIANT:
        return tuple(model.di_tilt_projection.parameters())
    raise AssertionError(variant)


def test_phase_c_adapters_receive_gradients_and_wake_up() -> None:
    architecture = _architecture()
    inputs = _inputs()
    rank = torch.linspace(-1.0, 1.0, 4)[None, :, None]
    targets = rank.expand(2, -1, 3).clone()
    label_mask = inputs["instrument_mask"][:, :4, None].expand(-1, -1, 3)

    for variant in ZERO_START_VARIANTS:
        torch.manual_seed(19)
        model = SharedCausalTCN(
            architecture=architecture, equity_count=4, variant=variant
        ).train()
        initial = [
            parameter.detach().clone()
            for parameter in _upstream_parameters(model, variant)
        ]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
        for _ in range(10):
            optimizer.zero_grad(set_to_none=True)
            loss = soft_spearman_loss(model(**inputs), targets, label_mask)
            loss.backward()
            assert torch.isfinite(loss)
            optimizer.step()
        upstream = _upstream_parameters(model, variant)
        assert any(
            not torch.equal(before, after)
            for before, after in zip(initial, upstream, strict=True)
        )


def _c2_summary(mean: float, fold_a: float, fold_b: float):
    return {
        "readouts": {
            "patience3_raw": {
                "mean_fold_candidate_minus_parent_ic": mean,
                "folds": {
                    "fold_a": {"candidate_minus_parent_primary_ic": fold_a},
                    "fold_b": {"candidate_minus_parent_primary_ic": fold_b},
                },
            }
        }
    }


def test_c2_extensions_require_threshold_and_nonnegative_folds() -> None:
    assert c2_extensions_allowed(_c2_summary(0.001, 0.002, 0.0))
    assert not c2_extensions_allowed(_c2_summary(0.00099, 0.002, 0.0))
    assert not c2_extensions_allowed(_c2_summary(0.001, 0.0021, -0.0001))


def test_phase_c_cli_requires_only_campaign_inputs() -> None:
    actions = {
        name
        for name, _ in parse_args(
            [
                "--parent-campaign",
                "parent",
                "--tilt-sidecar",
                "sidecar",
                "--output-dir",
                "output",
            ]
        )._get_kwargs()
    }
    assert actions == {"parent_campaign", "tilt_sidecar", "output_dir"}
