from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from brazil_rv.modeling import routing_identity_preflight as preflight
from brazil_rv.modeling.routing_identity_preflight import (
    PREFLIGHT_VERSION,
    compare_tensor_mappings,
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


def test_preflight_writes_failed_payload_before_failing_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        preflight,
        "validate_runtime",
        lambda: (_ for _ in ()).throw(RuntimeError("not a GH200")),
    )
    output = tmp_path / "routing_identity_preflight.json"
    with pytest.raises(RuntimeError, match="see"):
        run_routing_identity_preflight(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["version"] == PREFLIGHT_VERSION
    assert payload["status"] == "failed"
    assert payload["steps"] == 3
    assert payload["error"] == "RuntimeError: not a GH200"
    assert payload["optimizer"] == {"name": "sam_adamw", "rho": 0.125}
    assert payload["objective"] == {"name": "soft_spearman", "temperature": 0.5}
    assert not output.with_name(f"{output.name}.tmp").exists()
