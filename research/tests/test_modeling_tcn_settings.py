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
    TCN_FUSIONS,
    TCN_RECEPTIVE_FIELDS,
    TCN_WIDTHS,
    TCNSettings,
    expected_trainable_parameter_count,
    resolve_tcn_architecture,
)
from brazil_rv.modeling.model import (
    build_neural_model,
    count_trainable_parameters,
)


LEGAL_SETTINGS = tuple(
    TCNSettings(fusion, width, receptive_field)
    for fusion, width, receptive_field in product(
        TCN_FUSIONS, TCN_WIDTHS, TCN_RECEPTIVE_FIELDS
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
    assert dict(TCN_RECEPTIVE_FIELDS) == {
        "short": (1, 2, 4),
        "medium": (1, 2, 4, 8),
        "long": (1, 2, 4, 8, 16),
        "full": (1, 2, 4, 8, 16, 32),
    }
    for receptive_field, theoretical_patches in (
        ("short", 15),
        ("medium", 31),
        ("long", 63),
        ("full", 127),
    ):
        architecture = resolve_tcn_architecture(
            TCNSettings("none", 64, receptive_field)
        )
        assert architecture.dilations == TCN_RECEPTIVE_FIELDS[receptive_field]
        assert architecture.residual_blocks == len(architecture.dilations)
        assert architecture.theoretical_receptive_field_patches == theoretical_patches
        assert architecture.effective_receptive_field_patches == min(
            theoretical_patches, ABSOLUTE_PATCH_COUNT
        )
        assert (
            architecture.theoretical_receptive_field_minutes == 5 * theoretical_patches
        )
        assert architecture.effective_receptive_field_minutes == 5 * min(
            theoretical_patches, ABSOLUTE_PATCH_COUNT
        )


@pytest.mark.parametrize(
    "settings",
    LEGAL_SETTINGS,
    ids=lambda row: f"{row.fusion}-w{row.width}-{row.receptive_field}",
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


def test_tcn_parameter_counts_increase_strictly_with_width() -> None:
    for fusion, receptive_field in product(TCN_FUSIONS, TCN_RECEPTIVE_FIELDS):
        counts = [
            expected_trainable_parameter_count(
                "tcn",
                resolve_tcn_architecture(TCNSettings(fusion, width, receptive_field)),
            )
            for width in TCN_WIDTHS
        ]
        assert counts == sorted(counts)
        assert len(set(counts)) == len(counts)


def test_none_has_no_fusion_modules() -> None:
    model = _model(TCNSettings("none", 128, "full"))
    for name in ("fusion_input", "fusion_output", "fusion_gate", "fusion_norm"):
        assert not hasattr(model, name)
    assert not any(name.startswith("fusion") for name, _ in model.named_modules())


@pytest.mark.parametrize("fusion", TCN_FUSIONS)
def test_tcn_fusion_sources_are_exact_and_inactive_peers_are_masked(
    fusion: str,
) -> None:
    torch.manual_seed(23)
    model = _model(TCNSettings(fusion, 64, "short")).eval()
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
        TCNSettings("none", 64, "short"),
        TCNSettings("context_only", 128, "medium"),
        TCNSettings("pooled_market", 192, "long"),
        TCNSettings("context_pooled", 256, "full"),
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
    ("--tcn-fusion", "--tcn-width", "--tcn-receptive-field"),
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
def test_cli_forbids_tcn_settings_for_every_other_model(model_name: str) -> None:
    arguments = ["--model", model_name, "--seed", "11"]
    if model_name != "xgboost":
        arguments.extend(["--optimizer", "adamw", "--soft-rank-temperature", "0.10"])
    arguments.extend(
        [
            "--tcn-fusion",
            "none",
            "--tcn-width",
            "64",
            "--tcn-receptive-field",
            "short",
        ]
    )
    with pytest.raises(SystemExit):
        train.parse_args(arguments)


@pytest.mark.parametrize(
    "settings",
    (
        TCNSettings("attention", 64, "short"),
        TCNSettings("none", 96, "short"),
        TCNSettings("none", 64, "extra_full"),
    ),
)
def test_direct_tcn_configuration_rejects_invalid_values(
    settings: TCNSettings,
) -> None:
    with pytest.raises(ValueError):
        resolve_tcn_architecture(settings)
