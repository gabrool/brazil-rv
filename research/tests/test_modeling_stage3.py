from __future__ import annotations

import inspect

import pytest

from brazil_rv.modeling.model import (
    COMPRESSED_GLOBAL_RISK_VARIANT,
    DI_TILT_EXPOSURE_VARIANT,
    RESIDUAL_AUXILIARY_VARIANT,
)
from brazil_rv.modeling.stage3 import (
    experiment_numbering,
    parse_args,
    run_stage3,
    selected_stage3_variants,
)


def _phase_manifest() -> dict[str, object]:
    sequence = [
        RESIDUAL_AUXILIARY_VARIANT,
        COMPRESSED_GLOBAL_RISK_VARIANT,
        DI_TILT_EXPOSURE_VARIANT,
    ]
    return {
        "status": "completed",
        "official_validation_accessed": False,
        "test_accessed": False,
        "d2_training_gate_passed": True,
        "candidate_sequence": sequence,
        "results": {
            RESIDUAL_AUXILIARY_VARIANT: {"retained": False},
            COMPRESSED_GLOBAL_RISK_VARIANT: {"retained": True},
            DI_TILT_EXPOSURE_VARIANT: {"retained": False},
        },
    }


def _r1_summary(selected: str | None = None) -> dict[str, object]:
    return {
        "selected_candidate": selected,
        "official_validation_accessed": False,
        "test_accessed": False,
    }


def test_stage3_includes_gated_d2_and_only_retained_phase_c_variants() -> None:
    assert selected_stage3_variants(_phase_manifest(), _r1_summary()) == (
        RESIDUAL_AUXILIARY_VARIANT,
        COMPRESSED_GLOBAL_RISK_VARIANT,
    )


def test_stage3_rejects_unimplemented_nonnull_r1_recipe() -> None:
    with pytest.raises(ValueError, match="observed null R1"):
        selected_stage3_variants(_phase_manifest(), _r1_summary("blend"))


def test_stage3_experiment_numbers_follow_candidate_sequence() -> None:
    numbering = experiment_numbering(_phase_manifest())
    assert numbering["stage_0_and_1"] == {"D1": 15, "D2": 16, "R1": 17}
    assert numbering["candidate_numbers"] == {
        RESIDUAL_AUXILIARY_VARIANT: 18,
        COMPRESSED_GLOBAL_RISK_VARIANT: 19,
        DI_TILT_EXPOSURE_VARIANT: 20,
    }
    assert numbering["stage_3_number"] == 21


def test_stage3_has_no_test_or_split_control() -> None:
    assert tuple(inspect.signature(run_stage3).parameters) == (
        "store",
        "phase_c_campaign",
        "r1_summary_path",
        "d1_summary_path",
        "parent_reproduction",
        "selection_rule_file",
        "sidecar",
        "output_dir",
    )
    actions = {
        name
        for name, _ in parse_args(
            [
                "--phase-c-campaign",
                "phase-c",
                "--r1-summary",
                "r1.json",
                "--d1-summary",
                "d1.json",
                "--parent-reproduction",
                "parent",
                "--selection-rule-file",
                "selection.json",
                "--sidecar",
                "sidecar",
                "--output-dir",
                "output",
            ]
        )._get_kwargs()
    }
    assert actions == {
        "phase_c_campaign",
        "r1_summary",
        "d1_summary",
        "parent_reproduction",
        "selection_rule_file",
        "sidecar",
        "output_dir",
    }
