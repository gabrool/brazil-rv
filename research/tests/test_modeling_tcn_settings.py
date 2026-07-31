from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from itertools import product

import pytest
import torch

from brazil_rv.modeling import train
from brazil_rv.modeling.contract import (
    ABSOLUTE_PATCH_COUNT,
    BASELINE_TCN_SETTINGS,
    EQUITY_COUNT,
    INSTRUMENT_COUNT,
    PATCH_INPUT_WIDTH,
    SLOW_FEATURE_COUNT,
    TCN_BLOCK_VARIANTS,
    TCN_FUSIONS,
    TCN_RECEPTIVE_FIELDS,
    TCN_SWIGLU_HIDDEN_WIDTHS,
    TCN_WIDTHS,
    TCNSettings,
    expected_trainable_parameter_count,
    resolve_tcn_architecture,
)
from brazil_rv.modeling.layers import CausalTCNResidualBlock, SwiGLU
from brazil_rv.modeling.model import (
    build_neural_model,
    count_trainable_parameters,
)
from brazil_rv.modeling.optim import partition_parameters


LEGAL_SETTINGS = tuple(
    TCNSettings(fusion, width, receptive_field, block)
    for fusion, width, receptive_field, block in product(
        TCN_FUSIONS, TCN_WIDTHS, TCN_RECEPTIVE_FIELDS, TCN_BLOCK_VARIANTS
    )
)


def _model(settings: TCNSettings) -> torch.nn.Module:
    return build_neural_model("tcn", resolve_tcn_architecture(settings))


def _inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    patches = 0.1 * torch.randn(
        1,
        INSTRUMENT_COUNT,
        ABSOLUTE_PATCH_COUNT,
        PATCH_INPUT_WIDTH,
        generator=generator,
    )
    history = torch.zeros(1, INSTRUMENT_COUNT, ABSOLUTE_PATCH_COUNT, dtype=torch.bool)
    instrument = torch.zeros(1, INSTRUMENT_COUNT, dtype=torch.bool)
    instrument[:, :4] = True
    instrument[:, EQUITY_COUNT:] = True
    history[:, :4, 12:15] = True
    history[:, EQUITY_COUNT:, :15] = True
    slow = 0.1 * torch.randn(
        1, INSTRUMENT_COUNT, SLOW_FEATURE_COUNT, generator=generator
    )
    return {
        "patches": patches,
        "history_patch_mask": history,
        "instrument_mask": instrument,
        "slow_features": slow,
        "state_position": torch.tensor([15]),
    }


def _forward(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        inputs["patches"],
        inputs["history_patch_mask"],
        inputs["instrument_mask"],
        inputs["slow_features"],
        inputs["state_position"],
    )


def _changed(
    inputs: dict[str, torch.Tensor],
    instrument: int,
) -> dict[str, torch.Tensor]:
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["patches"][:, instrument] += 100.0
    changed["slow_features"][:, instrument] += 100.0
    return changed


def _update_digest(digest: object, name: str, tensor: torch.Tensor) -> None:
    digest.update(name.encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.detach().contiguous().numpy().tobytes())


def test_tcn_public_setting_and_receptive_field_contract() -> None:
    assert TCN_FUSIONS == (
        "none",
        "context_only",
        "pooled_market",
        "context_pooled",
    )
    assert TCN_WIDTHS == (64, 128, 192, 256)
    assert TCN_BLOCK_VARIANTS == ("gelu", "silu", "swiglu")
    assert dict(TCN_SWIGLU_HIDDEN_WIDTHS) == {64: 24, 128: 40, 192: 64, 256: 88}
    assert dict(TCN_RECEPTIVE_FIELDS) == {
        "short": (1, 1, 1, 1, 1, 2),
        "medium": (1, 2, 2, 2, 4, 4),
        "long": (1, 2, 4, 4, 4, 8),
        "full": (1, 2, 4, 8, 16, 32),
    }
    for receptive_field, theoretical_patches in (
        ("short", 15),
        ("medium", 31),
        ("long", 47),
        ("full", 127),
    ):
        architecture = resolve_tcn_architecture(
            TCNSettings("context_only", 64, receptive_field, "gelu")
        )
        assert architecture.dilations == TCN_RECEPTIVE_FIELDS[receptive_field]
        assert architecture.residual_blocks == 6
        assert architecture.theoretical_receptive_field_patches == theoretical_patches
        assert architecture.maximum_effective_equity_receptive_field_patches == min(
            theoretical_patches, 57
        )
        assert (
            architecture.theoretical_receptive_field_minutes == 5 * theoretical_patches
        )
        assert architecture.maximum_effective_equity_receptive_field_minutes == 5 * min(
            theoretical_patches, 57
        )
        assert architecture.maximum_effective_context_receptive_field_patches == min(
            theoretical_patches, ABSOLUTE_PATCH_COUNT
        )
        assert (
            architecture.maximum_effective_context_receptive_field_minutes
            == 5 * min(theoretical_patches, ABSOLUTE_PATCH_COUNT)
        )
    for fusion in ("none", "pooled_market"):
        architecture = resolve_tcn_architecture(TCNSettings(fusion, 64, "full", "gelu"))
        assert architecture.maximum_effective_context_receptive_field_patches is None
        assert architecture.maximum_effective_context_receptive_field_minutes is None


@pytest.mark.parametrize(
    "settings",
    LEGAL_SETTINGS,
    ids=lambda row: f"{row.fusion}-w{row.width}-{row.receptive_field}-{row.block}",
)
def test_every_tcn_setting_instantiates_with_exact_parameter_count(
    settings: TCNSettings,
) -> None:
    architecture = resolve_tcn_architecture(settings)
    model = _model(settings)
    assert architecture.fusion_width == 2 * settings.width
    assert (
        architecture.fusion_states
        == {
            "none": 0,
            "context_only": 7,
            "pooled_market": 3,
            "context_pooled": 9,
        }[settings.fusion]
    )
    assert count_trainable_parameters(model) == expected_trainable_parameter_count(
        "tcn", architecture
    )


def test_tcn_parameter_count_is_equal_across_receptive_fields() -> None:
    for fusion, width, block in product(TCN_FUSIONS, TCN_WIDTHS, TCN_BLOCK_VARIANTS):
        counts = {
            expected_trainable_parameter_count(
                "tcn",
                resolve_tcn_architecture(
                    TCNSettings(fusion, width, receptive_field, block)
                ),
            )
            for receptive_field in TCN_RECEPTIVE_FIELDS
        }
        assert len(counts) == 1


def test_tcn_parameter_counts_increase_strictly_with_width() -> None:
    for fusion, receptive_field, block in product(
        TCN_FUSIONS, TCN_RECEPTIVE_FIELDS, TCN_BLOCK_VARIANTS
    ):
        counts = [
            expected_trainable_parameter_count(
                "tcn",
                resolve_tcn_architecture(
                    TCNSettings(fusion, width, receptive_field, block)
                ),
            )
            for width in TCN_WIDTHS
        ]
        assert counts == sorted(counts)
        assert len(set(counts)) == len(counts)


def test_none_has_no_fusion_modules() -> None:
    model = _model(TCNSettings("none", 128, "full", "gelu"))
    for name in ("fusion_input", "fusion_output", "fusion_gate", "fusion_norm"):
        assert not hasattr(model, name)
    assert not any(name.startswith("fusion") for name, _ in model.named_modules())


@pytest.mark.parametrize(
    ("fusion", "block"), tuple(product(TCN_FUSIONS, TCN_BLOCK_VARIANTS))
)
def test_tcn_fusion_sources_are_exact_and_inactive_peers_are_masked(
    fusion: str, block: str
) -> None:
    torch.manual_seed(23)
    model = _model(TCNSettings(fusion, 64, "short", block)).eval()
    inputs = _inputs()
    with torch.no_grad():
        baseline = _forward(model, inputs)
        active_peer = _forward(model, _changed(inputs, 1))
        inactive_peer = _forward(model, _changed(inputs, 10))
        context = _forward(model, _changed(inputs, EQUITY_COUNT))
    peer_sensitive = fusion in ("pooled_market", "context_pooled")
    context_sensitive = fusion in ("context_only", "context_pooled")
    assert torch.equal(baseline[:, 0], active_peer[:, 0]) is not peer_sensitive
    assert torch.equal(baseline[:, 0], context[:, 0]) is not context_sensitive
    torch.testing.assert_close(
        baseline[:, :4], inactive_peer[:, :4], atol=0.0, rtol=0.0
    )
    inactive = ~inputs["instrument_mask"][:, :EQUITY_COUNT]
    assert torch.equal(baseline[inactive], torch.zeros_like(baseline[inactive]))


@pytest.mark.parametrize("fusion", ("context_only", "context_pooled"))
def test_unavailable_context_cannot_affect_tcn_output(fusion: str) -> None:
    torch.manual_seed(29)
    model = _model(TCNSettings(fusion, 64, "short", "gelu")).eval()
    inputs = _inputs()
    inputs["instrument_mask"][:, EQUITY_COUNT] = False
    changed = _changed(inputs, EQUITY_COUNT)
    with torch.no_grad():
        baseline = _forward(model, inputs)
        output = _forward(model, changed)
    torch.testing.assert_close(baseline, output, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("block", TCN_BLOCK_VARIANTS)
def test_tcn_block_structure_and_optimizer_routing(block: str) -> None:
    architecture = resolve_tcn_architecture(TCNSettings("none", 128, "short", block))
    model = _model(TCNSettings("none", 128, "short", block))
    for residual in model.blocks:
        if block == "swiglu":
            assert isinstance(residual.swiglu, SwiGLU)
            assert not hasattr(residual, "projection")
        else:
            assert hasattr(residual, "projection")
            assert not hasattr(residual, "swiglu")
    assert architecture.swiglu_hidden_width == (
        TCN_SWIGLU_HIDDEN_WIDTHS[128] if block == "swiglu" else None
    )
    groups = partition_parameters(model)
    routed_ids: set[int] = set()
    for parameters in groups.values():
        group_ids = {id(parameter) for parameter in parameters}
        assert routed_ids.isdisjoint(group_ids)
        routed_ids.update(group_ids)
    assert routed_ids == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


@pytest.mark.parametrize("block", ("gelu", "silu"))
def test_non_swiglu_block_rejects_hidden_width(block: str) -> None:
    with pytest.raises(
        ValueError,
        match="GELU and SiLU TCN blocks require swiglu_hidden_width=None",
    ):
        CausalTCNResidualBlock(64, 3, 1, 0.1, block, 24)


@pytest.mark.parametrize("hidden_width", (None, 0, -1, 1.5, True))
def test_swiglu_block_requires_positive_integer_hidden_width(
    hidden_width: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="SwiGLU TCN block requires a positive integer swiglu_hidden_width",
    ):
        CausalTCNResidualBlock(64, 3, 1, 0.1, "swiglu", hidden_width)


def test_residual_block_rejects_unknown_block() -> None:
    with pytest.raises(
        ValueError,
        match="TCN block must be one of: gelu, silu, swiglu",
    ):
        CausalTCNResidualBlock(64, 3, 1, 0.1, "relu", None)


def test_gelu_and_silu_parameter_counts_are_equal() -> None:
    for fusion, width, receptive_field in product(
        TCN_FUSIONS, TCN_WIDTHS, TCN_RECEPTIVE_FIELDS
    ):
        counts = [
            expected_trainable_parameter_count(
                "tcn",
                resolve_tcn_architecture(
                    TCNSettings(fusion, width, receptive_field, block)
                ),
            )
            for block in ("gelu", "silu")
        ]
        assert counts[0] == counts[1]


@pytest.mark.parametrize("block", TCN_BLOCK_VARIANTS)
def test_all_tcn_blocks_are_finite_masked_permutation_equivariant_and_causal(
    block: str,
) -> None:
    torch.manual_seed(37)
    model = _model(TCNSettings("context_pooled", 64, "full", block)).eval()
    inputs = _inputs()
    permutation = torch.arange(EQUITY_COUNT - 1, -1, -1)
    permuted = {key: value.clone() for key, value in inputs.items()}
    for key in ("patches", "history_patch_mask", "instrument_mask", "slow_features"):
        permuted[key][:, :EQUITY_COUNT] = inputs[key][:, :EQUITY_COUNT][:, permutation]
    future = {key: value.clone() for key, value in inputs.items()}
    future["patches"][:, :, 15:] += 1_000.0
    future["history_patch_mask"][:, :, 15:] = True
    with torch.no_grad():
        baseline = _forward(model, inputs)
        permuted_output = _forward(model, permuted)
        future_output = _forward(model, future)
    assert baseline.shape == (1, EQUITY_COUNT, 3)
    assert torch.isfinite(baseline).all()
    torch.testing.assert_close(
        permuted_output, baseline[:, permutation], atol=3e-5, rtol=3e-5
    )
    torch.testing.assert_close(future_output, baseline, atol=0.0, rtol=0.0)
    inactive = ~inputs["instrument_mask"][:, :EQUITY_COUNT]
    assert torch.equal(baseline[inactive], torch.zeros_like(baseline[inactive]))


def test_baseline_tcn_state_layout_count_and_seeded_output_are_exact() -> None:
    torch.manual_seed(1234)
    architecture = resolve_tcn_architecture(BASELINE_TCN_SETTINGS)
    model = _model(BASELINE_TCN_SETTINGS).eval()
    expected_keys = ["input_projection.weight"]
    for block in range(6):
        expected_keys.extend(
            (
                f"blocks.{block}.convolution.weight",
                f"blocks.{block}.convolution.bias",
                f"blocks.{block}.norm.weight",
                f"blocks.{block}.norm.bias",
                f"blocks.{block}.projection.weight",
            )
        )
    expected_keys.extend(
        (
            "slow_projection.weight",
            "state_norm.weight",
            "state_norm.bias",
            "fusion_input.weight",
            "fusion_input.bias",
            "fusion_output.weight",
            "fusion_gate.weight",
            "fusion_gate.bias",
            "fusion_norm.weight",
            "fusion_norm.bias",
            "prediction_head.weight",
            "prediction_head.bias",
        )
    )
    assert tuple(model.state_dict()) == tuple(expected_keys)
    assert count_trainable_parameters(model) == 777_987
    assert expected_trainable_parameter_count("tcn", architecture) == 777_987

    state_digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        _update_digest(state_digest, key, value)
    assert (
        state_digest.hexdigest()
        == "cce6cfeff5335f7275819747043a7962191157e3c43d2ac49e4cbcfe55da45bf"
    )

    generator = torch.Generator().manual_seed(5678)
    patches = torch.randn(
        1,
        INSTRUMENT_COUNT,
        ABSOLUTE_PATCH_COUNT,
        PATCH_INPUT_WIDTH,
        generator=generator,
    )
    history = torch.ones(1, INSTRUMENT_COUNT, ABSOLUTE_PATCH_COUNT, dtype=torch.bool)
    instrument = torch.ones(1, INSTRUMENT_COUNT, dtype=torch.bool)
    instrument[:, 4:EQUITY_COUNT] = False
    slow = torch.randn(1, INSTRUMENT_COUNT, SLOW_FEATURE_COUNT, generator=generator)
    with torch.no_grad():
        output = model(patches, history, instrument, slow, torch.tensor([69]))
    output_digest = hashlib.sha256()
    _update_digest(output_digest, "output", output)
    assert (
        output_digest.hexdigest()
        == "59ed77569ca2d6b42f543ae09b04a157cda2e8dd986fe63edb5b86aaf785afff"
    )


@pytest.mark.parametrize(
    "settings",
    (
        TCNSettings("none", 64, "short", "gelu"),
        TCNSettings("context_only", 128, "medium", "silu"),
        TCNSettings("pooled_market", 192, "long", "swiglu"),
    ),
)
def test_selected_tcn_state_dict_round_trip(settings: TCNSettings) -> None:
    torch.manual_seed(19)
    model = _model(settings).eval()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = _model(settings).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert model.state_dict().keys() == restored.state_dict().keys()
    for key, expected in model.state_dict().items():
        torch.testing.assert_close(
            restored.state_dict()[key], expected, atol=0.0, rtol=0.0
        )


def test_run_names_and_manifests_distinguish_all_tcn_settings() -> None:
    created_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    names = []
    identities = []
    for settings in LEGAL_SETTINGS:
        architecture = resolve_tcn_architecture(settings)
        names.append(
            train._run_directory_name(
                "tcn",
                settings,
                "adamw",
                "soft_spearman",
                0.1,
                None,
                11,
                created_at,
            )
        )
        identities.append(
            json.dumps(
                train._model_metadata("tcn", architecture, settings),
                sort_keys=True,
            )
        )
    assert len(set(names)) == len(LEGAL_SETTINGS)
    assert len(set(identities)) == len(LEGAL_SETTINGS)


def _tcn_cli(settings: TCNSettings) -> list[str]:
    return [
        "--model",
        "tcn",
        "--tcn-block",
        settings.block,
        "--tcn-fusion",
        settings.fusion,
        "--tcn-width",
        str(settings.width),
        "--tcn-receptive-field",
        settings.receptive_field,
        "--optimizer",
        "adamw",
        "--soft-rank-temperature",
        "0.10",
        "--seed",
        "11",
    ]


def test_cli_accepts_every_legal_tcn_setting() -> None:
    for settings in LEGAL_SETTINGS:
        args = train.parse_args(_tcn_cli(settings))
        assert train._tcn_settings_from_args(args) == settings


@pytest.mark.parametrize(
    "flag",
    ("--tcn-fusion", "--tcn-width", "--tcn-receptive-field", "--tcn-block"),
)
def test_cli_requires_all_tcn_settings(flag: str) -> None:
    arguments = _tcn_cli(BASELINE_TCN_SETTINGS)
    index = arguments.index(flag)
    del arguments[index : index + 2]
    with pytest.raises(SystemExit):
        train.parse_args(arguments)


@pytest.mark.parametrize(
    ("flag", "value"),
    (
        ("--tcn-block", "relu"),
        ("--tcn-fusion", "attention"),
        ("--tcn-width", "96"),
        ("--tcn-receptive-field", "extra_full"),
    ),
)
def test_cli_rejects_invalid_tcn_values(flag: str, value: str) -> None:
    arguments = _tcn_cli(BASELINE_TCN_SETTINGS)
    arguments[arguments.index(flag) + 1] = value
    with pytest.raises(SystemExit):
        train.parse_args(arguments)


@pytest.mark.parametrize(
    "model_name",
    (
        "temporal_only",
        "context_only",
        "pooled_market",
        "context_pooled",
        "mlp",
        "xgboost",
    ),
)
def test_cli_forbids_tcn_settings_for_every_other_model(
    model_name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = ["--model", model_name, "--seed", "11"]
    if model_name != "xgboost":
        arguments.extend(["--optimizer", "adamw", "--soft-rank-temperature", "0.10"])
    arguments.extend(
        [
            "--tcn-fusion",
            "none",
            "--tcn-width",
            "64",
            "--tcn-block",
            "gelu",
            "--tcn-receptive-field",
            "short",
        ]
    )
    with pytest.raises(SystemExit):
        train.parse_args(arguments)
    assert (
        "TCN architecture arguments are forbidden unless --model tcn"
        in capsys.readouterr().err
    )


@pytest.mark.parametrize(
    "settings",
    (
        TCNSettings("attention", 64, "short", "gelu"),
        TCNSettings("none", 96, "short", "gelu"),
        TCNSettings("none", 64, "extra_full", "gelu"),
        TCNSettings("none", 64, "short", "relu"),
    ),
)
def test_direct_tcn_configuration_rejects_invalid_values(
    settings: TCNSettings,
) -> None:
    with pytest.raises(ValueError):
        resolve_tcn_architecture(settings)
