from dataclasses import replace

import pytest

from brazil_rv.modeling.contract import (
    DYNAMIC_CHANNEL_COUNT,
    TCN_ARCHITECTURE,
    TrainingSpecification,
)
from brazil_rv.modeling.hpo_sweep import sweep_configurations


def test_experiment47_roster_and_frozen_cells_are_exact() -> None:
    configs = {config.config_id: config for config in sweep_configurations()}
    assert tuple(configs) == (
        "C0",
        "S1",
        "S2",
        "S3",
        "P1",
        "P2",
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6",
        "R7",
        "R8",
        "R9",
        "R10",
    )
    assert configs["C0"].specification == TrainingSpecification()
    assert configs["R1"].receptive_field == 107
    assert configs["R2"].receptive_field == 131
    assert configs["R3"].receptive_field == 85
    assert configs["R9"].receptive_field == 91
    assert configs["R10"].receptive_field == 59
    assert configs["R4"].specification.patch_minutes == 10
    assert (
        configs["R4"].specification.architecture.patch_input_width
        == 10 * DYNAMIC_CHANNEL_COUNT
    )
    assert configs["S1"].retained_component_count == 6
    assert configs["R1"].retained_component_count == 6


def test_training_specification_rejects_patch_width_mismatch() -> None:
    with pytest.raises(ValueError, match="input width"):
        TrainingSpecification(
            architecture=replace(TCN_ARCHITECTURE, patch_input_width=260),
            patch_minutes=5,
        )
