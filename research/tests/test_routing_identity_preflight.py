from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from brazil_rv.modeling.contract import EQUITY_COUNT, GH200_RUNTIME, LOCAL_CONTEXT_COUNT
from brazil_rv.modeling import routing_identity_preflight as preflight
from brazil_rv.modeling.routing_identity_preflight import (
    build_routing_preflight_identity,
    PREFLIGHT_VERSION,
    compare_tensor_mappings,
    validate_routing_identity_preflight,
    run_routing_identity_preflight,
)


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
        return {
            **base,
            "legacy": {"gradient_norm": 1.0},
            "scaffold": {"gradient_norm": 1.0},
        }
    if quantity == "base_parameter_coverage":
        names = [row["name"] for row in identity["parameter_identities"]["base"]]
        report = {
            "per_parameter": {
                name: {
                    "first_pass_gradient": True,
                    "second_pass_gradient": True,
                    "parameter_update": True,
                }
                for name in names
            },
            "legitimate_zero_gradient_exceptions": [],
            "exception_policy": "No base parameter is expected to have zero gradients or updates.",
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


def test_complete_passed_artifact_is_reusable(
    expected_identity: dict[str, object], environment: dict[str, object]
) -> None:
    payload = _passed_payload(expected_identity, environment)
    assert (
        validate_routing_identity_preflight(payload, expected_identity, environment)
        is payload
    )


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
