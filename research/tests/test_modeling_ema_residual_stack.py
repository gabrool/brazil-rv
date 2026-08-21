from __future__ import annotations

import inspect
import json

import pytest

from brazil_rv.modeling import ema_residual_stack
from brazil_rv.modeling.ema_residual_stack import (
    DISCOVERY_MARGIN,
    discovery_gate_passed,
    parse_args,
    run_ema_residual_stack_reanalysis,
)


def test_discovery_gate_requires_point_zero_zero_one_on_each_fold() -> None:
    assert DISCOVERY_MARGIN == 0.001
    assert discovery_gate_passed({"fold_a": 0.001, "fold_b": 0.0011})
    assert not discovery_gate_passed({"fold_a": 0.00099, "fold_b": 0.01})
    assert not discovery_gate_passed({"fold_a": 0.01, "fold_b": -0.001})
    with pytest.raises(ValueError, match="exactly fold_a and fold_b"):
        discovery_gate_passed({"fold_a": 0.01})


def test_driver_has_no_test_or_split_control() -> None:
    assert tuple(inspect.signature(run_ema_residual_stack_reanalysis).parameters) == (
        "parent_campaign",
        "residual_campaign",
        "parent_reproduction",
        "official_campaign",
        "output_dir",
    )
    actions = {
        name
        for name, _ in parse_args(
            [
                "--parent-campaign",
                "parent",
                "--residual-campaign",
                "residual",
                "--parent-reproduction",
                "reproduction",
                "--official-campaign",
                "official",
                "--output-dir",
                "output",
            ]
        )._get_kwargs()
    }
    assert actions == {
        "parent_campaign",
        "residual_campaign",
        "parent_reproduction",
        "official_campaign",
        "output_dir",
    }


def test_failed_discovery_gate_never_opens_official_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(ema_residual_stack, "repository_commit", lambda: "abc123")
    monkeypatch.setattr(
        ema_residual_stack,
        "_run_discovery",
        lambda *_: {"gate": {"passed": False, "fold_deltas": {}}},
    )

    def fail_if_called(*_):
        raise AssertionError("official validation was opened behind a failed gate")

    monkeypatch.setattr(ema_residual_stack, "_run_official", fail_if_called)
    output = tmp_path / "output"
    run_ema_residual_stack_reanalysis(
        tmp_path / "parent",
        tmp_path / "residual",
        tmp_path / "reproduction",
        tmp_path / "official",
        output,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["official_validation_accessed"] is False
    assert manifest["official"] is None
    assert manifest["test_accessed"] is False
