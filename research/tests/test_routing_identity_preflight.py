from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from brazil_rv.modeling.contract import EQUITY_COUNT, GH200_RUNTIME, LOCAL_CONTEXT_COUNT
from brazil_rv.modeling.engine import soft_spearman_loss
from brazil_rv.modeling.run_profiles import RUN_PROFILE_SCHEMA_VERSION
from brazil_rv.modeling import routing_identity_preflight as preflight
from brazil_rv.modeling.routing_identity_preflight import (
    build_routing_preflight_identity,
    PREFLIGHT_VERSION,
    compare_tensor_mappings,
    validate_routing_identity_preflight,
    run_routing_identity_preflight,
)


def _previous_non_scalar_tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_tensor_hash_supports_zero_dimensional_floating_scalars(
    dtype: torch.dtype,
) -> None:
    scalar = torch.tensor(1.25, dtype=dtype)
    clone = scalar.clone()
    changed = torch.tensor(1.5, dtype=dtype)

    assert scalar.shape == torch.Size([])
    assert preflight._tensor_hash(scalar) == preflight._tensor_hash(clone)
    assert preflight._tensor_hash(scalar) != preflight._tensor_hash(changed)


def test_mapping_hash_supports_scalar_and_non_scalar_tensors() -> None:
    values = {
        "gradient_norm": torch.tensor(0.75, dtype=torch.float32),
        "weights": torch.arange(6, dtype=torch.float32).reshape(2, 3),
    }
    clone = {name: value.clone() for name, value in values.items()}
    changed = {**clone, "gradient_norm": torch.tensor(0.5)}

    assert preflight._mapping_hash(values) == preflight._mapping_hash(clone)
    assert preflight._mapping_hash(values) != preflight._mapping_hash(changed)


@pytest.mark.parametrize(
    "tensor",
    (
        torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32),
        torch.arange(12, dtype=torch.int64).reshape(3, 4),
        torch.empty(0, dtype=torch.float32),
    ),
)
def test_tensor_hash_preserves_previous_non_scalar_bytes(tensor: torch.Tensor) -> None:
    assert preflight._tensor_hash(tensor) == _previous_non_scalar_tensor_hash(tensor)


def test_tensor_comparison_records_hashes_and_maximum_errors() -> None:
    left = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([-3.0])}
    exact = compare_tensor_mappings(
        left, {key: value.clone() for key, value in left.items()}, atol=0.0, rtol=0.0
    )
    assert exact["passed"] is True
    assert exact["maximum_absolute_error"] == 0.0
    assert exact["left_sha256"] == exact["right_sha256"]

    changed = {"a": torch.tensor([1.0, 2.1]), "b": torch.tensor([-3.0])}
    mismatch = compare_tensor_mappings(left, changed, atol=1e-3, rtol=0.0)
    assert mismatch["passed"] is False
    assert mismatch["maximum_absolute_error"] == pytest.approx(0.1)
    assert mismatch["relative_l2_error"] > 0.0
    assert mismatch["left_sha256"] != mismatch["right_sha256"]

    identity = compare_tensor_mappings(
        left, {"different": torch.ones(1)}, atol=0.0, rtol=0.0
    )
    assert identity == {
        "passed": False,
        "reason": "tensor_identity_mismatch",
        "left_names": ["a", "b"],
        "right_names": ["different"],
    }


@pytest.fixture(scope="module")
def expected_identity() -> dict[str, object]:
    return build_routing_preflight_identity("a" * 40)


@pytest.fixture(scope="module")
def environment() -> dict[str, object]:
    return {
        "device_name": "NVIDIA GH200 480GB",
        "compute_capability": [9, 0],
        "total_vram_bytes": GH200_RUNTIME.minimum_vram_bytes,
        "cpu_architecture": "aarch64",
        "platform": "Linux synthetic",
        "pytorch_version": "2.13.0",
        "cuda_version": "12.6",
        "cudnn_version": 90100,
    }


def _sam_diagnostics(**overrides: object) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "all_finite": True,
        "backward_passes": 16,
        "bounded_cuda_phase_seconds": None,
        "first_pass_gradient_norm": 6.351141929626465,
        "gradient_norm": 6.351141929626465,
        "h2d_bytes": 4096,
        "h2d_enqueue_seconds": 0.001,
        "loss_count": 8,
        "loss_sum": 1.25,
        "perturbation_norm": 0.125,
        "predictions_finite": True,
        "rng_replay_exact": True,
        "second_loss_sum": 1.5,
        "second_pass_gradient_norm": 6.351141929626465,
        "total_effective_update_wall_seconds": 0.01,
    }
    diagnostics.update(overrides)
    return diagnostics


def _comparison(
    execution: str,
    step: int,
    quantity: str,
    identity: dict[str, object],
) -> dict[str, object]:
    base = {"execution": execution, "step": step, "quantity": quantity, "passed": True}
    if quantity in preflight._TENSOR_QUANTITIES:
        return {
            **base,
            "maximum_absolute_error": 0.0,
            "relative_l2_error": 0.0,
            "left_sha256": "0" * 64,
            "right_sha256": "0" * 64,
            "tensor_count": 1,
        }
    if quantity in ("prediction_rng_state", "update_rng_state"):
        return {**base, "left_sha256": "1" * 64, "right_sha256": "1" * 64}
    if quantity == "sam_scalar_diagnostics":
        legacy = _sam_diagnostics()
        scaffold = _sam_diagnostics(
            gradient_norm=6.351142406463623,
            second_pass_gradient_norm=6.351142406463623,
            h2d_enqueue_seconds=0.002,
            total_effective_update_wall_seconds=0.02,
        )
        return {
            **base,
            **preflight._compare_sam_scalar_diagnostics(legacy, scaffold),
        }
    if quantity == "base_parameter_coverage":
        names = [row["name"] for row in identity["parameter_identities"]["base"]]
        exceptions = set(preflight._ZERO_GRADIENT_EXCEPTIONS)
        report = {
            "per_parameter": {
                name: {
                    "first_pass_gradient": name not in exceptions,
                    "second_pass_gradient": name not in exceptions,
                    "parameter_update": name not in exceptions,
                }
                for name in names
            },
            "legitimate_zero_gradient_exceptions": [
                {"name": name, "justification": justification}
                for name, justification in sorted(
                    preflight._ZERO_GRADIENT_EXCEPTIONS.items()
                )
            ],
            "exception_policy": "Only the explicitly justified parameters may remain unexercised.",
            "unexercised_expected_parameters": [],
            "passed": True,
        }
        return {
            **base,
            "expected_parameter_names": names,
            "legacy": deepcopy(report),
            "scaffold": deepcopy(report),
        }
    if quantity == "inactive_routing_parameters":
        names = [row["name"] for row in identity["parameter_identities"]["routing"]]
        return {
            **base,
            "expected_parameter_names": names,
            "gradient_parameter_names": [],
            "changed_parameter_names": [],
            "sha256": "2" * 64,
        }
    raise AssertionError(quantity)


def _passed_payload(
    identity: dict[str, object], environment: dict[str, object]
) -> dict[str, object]:
    runtime = identity["runtime"]
    compile_setup = {
        "api": "nn.Module.compile",
        "backend": runtime["compile_backend"],
        "mode": runtime["compile_mode"],
        "fullgraph": runtime["compile_fullgraph"],
        "dynamic": runtime["compile_dynamic"],
        "backward_pass_autocast_control_available": True,
        "backward_pass_autocast_policy": "explicit_off",
    }
    pairs = [
        (row["step"], row["quantity"])
        for row in identity["execution_contract"]["required_comparisons"]
    ]
    executions = []
    for mode in preflight.PREFLIGHT_EXECUTIONS:
        comparisons = [
            _comparison(mode, step, quantity, identity) for step, quantity in pairs
        ]
        executions.append(
            {
                "execution": mode,
                "compile_setups": (
                    {}
                    if mode == "eager"
                    else {"legacy": compile_setup, "scaffold": deepcopy(compile_setup)}
                ),
                "comparisons": comparisons,
                "passed": True,
            }
        )
    return {
        "version": PREFLIGHT_VERSION,
        "status": "passed",
        "steps": 3,
        "seed": preflight.PREFLIGHT_SEED,
        "identity": deepcopy(identity),
        "identity_sha256": preflight._json_hash(identity),
        "objective": deepcopy(identity["objective"]),
        "optimizer": deepcopy(identity["optimizer"]),
        "sam": deepcopy(identity["sam"]),
        "runtime": deepcopy(identity["runtime"]),
        "tolerances": deepcopy(preflight.PREFLIGHT_TOLERANCES),
        "synthetic_fixture": deepcopy(identity["synthetic_fixture"]),
        "environment": deepcopy(environment),
        "executions": executions,
        "maximum_observed_errors": {"absolute": 0.0, "relative_l2": 0.0},
    }


def _sam_comparisons(payload: dict[str, object]) -> list[dict[str, object]]:
    return [
        comparison
        for execution in payload["executions"]
        for comparison in execution["comparisons"]
        if comparison["quantity"] == "sam_scalar_diagnostics"
    ]


def _coverage_comparisons(payload: dict[str, object]) -> list[dict[str, object]]:
    return [
        comparison
        for execution in payload["executions"]
        for comparison in execution["comparisons"]
        if comparison["quantity"] == "base_parameter_coverage"
    ]


def test_sam_diagnostics_accept_timing_differences_and_observed_norm_delta() -> None:
    legacy = _sam_diagnostics()
    scaffold = _sam_diagnostics(
        gradient_norm=6.351142406463623,
        second_pass_gradient_norm=6.351142406463623,
        h2d_enqueue_seconds=10.0,
        total_effective_update_wall_seconds=20.0,
    )

    comparison = preflight._compare_sam_scalar_diagnostics(legacy, scaffold)

    assert comparison["passed"] is True
    assert comparison["legacy"] is legacy
    assert comparison["scaffold"] is scaffold


def test_sam_diagnostics_accept_different_valid_bounded_cuda_timings() -> None:
    legacy_timings = {
        phase: float(index + 1) / 1000
        for index, phase in enumerate(preflight.SAM_BOUNDED_CUDA_PHASES)
    }
    scaffold_timings = {
        phase: float(index + 1) / 10
        for index, phase in enumerate(preflight.SAM_BOUNDED_CUDA_PHASES)
    }

    comparison = preflight._compare_sam_scalar_diagnostics(
        _sam_diagnostics(bounded_cuda_phase_seconds=legacy_timings),
        _sam_diagnostics(bounded_cuda_phase_seconds=scaffold_timings),
    )

    assert comparison["passed"] is True


def test_sam_diagnostics_reject_deterministic_and_continuous_drift() -> None:
    deterministic = preflight._compare_sam_scalar_diagnostics(
        _sam_diagnostics(), _sam_diagnostics(loss_count=9)
    )
    continuous = preflight._compare_sam_scalar_diagnostics(
        _sam_diagnostics(), _sam_diagnostics(gradient_norm=6.36)
    )

    assert deterministic["passed"] is False
    assert continuous["passed"] is False


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("missing", None),
        ("extra", None),
        ("loss_count", 8.0),
        ("loss_sum", float("nan")),
        ("gradient_norm", float("inf")),
        ("h2d_enqueue_seconds", -0.001),
        ("total_effective_update_wall_seconds", -0.001),
        (
            "bounded_cuda_phase_seconds",
            {phase: 0.001 for phase in preflight.SAM_BOUNDED_CUDA_PHASES},
        ),
    ),
)
def test_sam_diagnostic_malformed_records_fail_closed(
    mutation: str,
    value: object,
    expected_identity: dict[str, object],
    environment: dict[str, object],
) -> None:
    payload = _passed_payload(expected_identity, environment)
    scaffold = _sam_comparisons(payload)[0]["scaffold"]
    if mutation == "missing":
        scaffold.pop("loss_sum")
    elif mutation == "extra":
        scaffold["unexpected"] = 1
    else:
        scaffold[mutation] = value

    with pytest.raises(ValueError, match="stored SAM diagnostics status"):
        validate_routing_identity_preflight(payload, expected_identity, environment)


def test_sam_diagnostic_forged_passed_status_is_rejected(
    expected_identity: dict[str, object], environment: dict[str, object]
) -> None:
    payload = _passed_payload(expected_identity, environment)
    comparison = _sam_comparisons(payload)[0]
    comparison["scaffold"]["rng_replay_exact"] = False
    comparison["passed"] = True

    with pytest.raises(ValueError, match="stored SAM diagnostics status"):
        validate_routing_identity_preflight(payload, expected_identity, environment)


def test_soft_spearman_structural_bias_null_directions() -> None:
    torch.manual_seed(17)
    fusion_norm = nn.LayerNorm(7)
    prediction_head = nn.Linear(7, 3)
    states = torch.randn(2, 11, 7)
    targets = torch.linspace(-0.9, 0.9, 11).reshape(1, 11, 1).expand(2, -1, 3)
    mask = torch.ones_like(targets, dtype=torch.bool)

    predictions = prediction_head(fusion_norm(states))
    shifted = soft_spearman_loss(
        predictions + torch.tensor([3.0, -5.0, 11.0]), targets, mask, 0.5
    )
    loss = soft_spearman_loss(predictions, targets, mask, 0.5)
    loss.backward()

    torch.testing.assert_close(shifted, loss, atol=2e-6, rtol=0.0)
    assert prediction_head.bias.grad is not None
    assert fusion_norm.bias.grad is not None
    assert float(prediction_head.bias.grad.abs().max()) < 2e-6
    assert float(fusion_norm.bias.grad.abs().max()) < 2e-6
    assert float(prediction_head.weight.grad.abs().max()) > 1e-5
    assert float(fusion_norm.weight.grad.abs().max()) > 1e-5


def test_preflight_identity_binds_experiment_profile_and_packed_shape() -> None:
    profile = {
        "schema_version": RUN_PROFILE_SCHEMA_VERSION,
        "name": "experiment",
        "equity_count": 48,
        "equity_slots": list(range(48)),
        "decision_indices": list(range(0, 55, 3)),
        "maximum_epochs": 3,
        "identity_sha256": "profile-identity",
    }
    identity = build_routing_preflight_identity("a" * 40, profile, 9_728)

    assert identity["run_profile"] == profile
    assert identity["packed_shape"] == {
        "equity_count": 48,
        "instrument_count": 63,
    }
    assert identity["optimizer"]["train_sample_count"] == 9_728
    assert identity["optimizer"]["maximum_epochs"] == 3
    assert identity["synthetic_fixture"]["packed_equity_count"] == 48
    assert identity["synthetic_fixture"]["packed_instrument_count"] == 63
    assert identity["synthetic_fixture"]["valid_nonzero_values"] > 0
    assert identity["synthetic_fixture"]["unavailable_nonzero_values"] == 0


def test_approved_null_directions_remain_in_identity_and_parity() -> None:
    identity = build_routing_preflight_identity("a" * 40)
    expected_exceptions = set(preflight._ZERO_GRADIENT_EXCEPTIONS)
    identity_names = {row["name"] for row in identity["parameter_identities"]["base"]}
    legacy, scaffold = preflight._seeded_models("cpu")
    legacy_base = preflight._parameter_snapshot(legacy, routing=False)
    scaffold_base = preflight._parameter_snapshot(scaffold, routing=False)
    for name, parameter in legacy.named_parameters():
        if not name.startswith("routing."):
            parameter.grad = torch.zeros_like(parameter)
    gradients = preflight._gradient_snapshot(legacy, routing=False)
    updates = preflight._difference_snapshot(legacy_base, legacy_base)
    comparison = compare_tensor_mappings(legacy_base, scaffold_base, atol=0.0, rtol=0.0)
    changed = {name: value.clone() for name, value in legacy_base.items()}
    changed["prediction_head.bias"].add_(1.0)

    assert expected_exceptions == {"fusion_norm.bias", "prediction_head.bias"}
    assert expected_exceptions <= identity_names
    assert expected_exceptions <= legacy_base.keys()
    assert comparison["passed"] is True
    assert comparison["tensor_count"] == len(legacy_base)
    assert expected_exceptions <= gradients.keys()
    assert expected_exceptions <= updates.keys()
    assert preflight._mapping_hash(changed) != preflight._mapping_hash(legacy_base)


def test_artifact_with_only_approved_null_directions_is_reusable(
    expected_identity: dict[str, object], environment: dict[str, object]
) -> None:
    payload = _passed_payload(expected_identity, environment)
    assert (
        validate_routing_identity_preflight(payload, expected_identity, environment)
        is payload
    )


def test_unapproved_unexercised_parameter_is_rejected(
    expected_identity: dict[str, object], environment: dict[str, object]
) -> None:
    payload = _passed_payload(expected_identity, environment)
    coverage = _coverage_comparisons(payload)[0]["legacy"]
    name = next(
        name
        for name in coverage["per_parameter"]
        if name not in preflight._ZERO_GRADIENT_EXCEPTIONS
    )
    coverage["per_parameter"][name]["first_pass_gradient"] = False
    coverage["unexercised_expected_parameters"] = [name]
    coverage["passed"] = False
    with pytest.raises(
        ValueError, match="contains unexercised expected base parameters"
    ):
        validate_routing_identity_preflight(payload, expected_identity, environment)


@pytest.mark.parametrize("declaration_drift", ["missing", "extra", "modified"])
def test_exception_declaration_drift_is_rejected(
    declaration_drift: str,
    expected_identity: dict[str, object],
    environment: dict[str, object],
) -> None:
    payload = _passed_payload(expected_identity, environment)
    declarations = _coverage_comparisons(payload)[0]["legacy"][
        "legitimate_zero_gradient_exceptions"
    ]
    if declaration_drift == "missing":
        declarations.pop()
    elif declaration_drift == "extra":
        declarations.append(
            {"name": "unexpected.bias", "justification": "not approved"}
        )
    else:
        declarations[0]["justification"] += " altered"
    with pytest.raises(ValueError, match="zero-gradient exceptions drifted"):
        validate_routing_identity_preflight(payload, expected_identity, environment)


def test_complete_passed_artifact_is_reusable(
    expected_identity: dict[str, object], environment: dict[str, object]
) -> None:
    payload = _passed_payload(expected_identity, environment)
    assert (
        validate_routing_identity_preflight(payload, expected_identity, environment)
        is payload
    )


def test_previous_preflight_version_is_not_reusable(
    expected_identity: dict[str, object], environment: dict[str, object]
) -> None:
    payload = _passed_payload(expected_identity, environment)
    payload["version"] = "ROUTING_IDENTITY_PREFLIGHT_V1"
    with pytest.raises(ValueError, match="version mismatch"):
        validate_routing_identity_preflight(payload, expected_identity, environment)


def test_minimal_passed_payload_is_rejected(
    expected_identity: dict[str, object], environment: dict[str, object]
) -> None:
    minimal = {"version": PREFLIGHT_VERSION, "status": "passed", "steps": 3}
    with pytest.raises(ValueError, match="top-level artifact"):
        validate_routing_identity_preflight(minimal, expected_identity, environment)


def test_previous_commit_payload_is_rejected(environment: dict[str, object]) -> None:
    previous = build_routing_preflight_identity("b" * 40)
    current = build_routing_preflight_identity("a" * 40)
    with pytest.raises(ValueError, match="expected identity mismatch"):
        validate_routing_identity_preflight(
            _passed_payload(previous, environment), current, environment
        )


def test_missing_comparison_is_rejected(
    expected_identity: dict[str, object], environment: dict[str, object]
) -> None:
    payload = _passed_payload(expected_identity, environment)
    payload["executions"][0]["comparisons"].pop()
    with pytest.raises(ValueError, match="required comparisons"):
        validate_routing_identity_preflight(payload, expected_identity, environment)


@pytest.mark.parametrize("drift", ["environment", "configuration"])
def test_environment_and_configuration_drift_are_rejected(
    drift: str,
    expected_identity: dict[str, object],
    environment: dict[str, object],
) -> None:
    payload = _passed_payload(expected_identity, environment)
    if drift == "environment":
        payload["environment"]["pytorch_version"] = "drifted"
        match = "environment drifted"
    else:
        payload["optimizer"]["lr"] = 0.0
        match = "optimizer drifted"
    with pytest.raises(ValueError, match=match):
        validate_routing_identity_preflight(payload, expected_identity, environment)


def test_temporal_fixture_is_nonzero_only_at_causally_available_sources() -> None:
    batch = preflight._synthetic_batch()
    fixture = preflight._fixture_metadata(batch)
    assert fixture["finite"] is True
    assert fixture["valid_nonzero_values"] == fixture["valid_patch_positions"] * 130
    assert fixture["unavailable_nonzero_values"] == 0
    assert fixture["source_valid_patch_positions"] == {
        "equity": 96,
        "wdo": 15,
        "di": 75,
        "zt": 69,
        "zn": 69,
    }
    assert not batch["instrument_mask"][0, EQUITY_COUNT]
    assert not batch["instrument_mask"][0, EQUITY_COUNT + LOCAL_CONTEXT_COUNT]


def test_preflight_writes_failed_payload_before_failing_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        preflight,
        "validate_runtime",
        lambda: (_ for _ in ()).throw(RuntimeError("not a GH200")),
    )
    output = tmp_path / "routing_identity_preflight.json"
    identity = build_routing_preflight_identity("a" * 40)
    with pytest.raises(RuntimeError, match="see"):
        run_routing_identity_preflight(output, identity)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["version"] == PREFLIGHT_VERSION
    assert payload["status"] == "failed"
    assert payload["steps"] == 3
    assert payload["error"] == "RuntimeError: not a GH200"
    assert payload["optimizer"] == identity["optimizer"]
    assert payload["objective"] == identity["objective"]
    assert payload["sam"] == identity["sam"]
    assert not output.with_name(f"{output.name}.tmp").exists()
