from dataclasses import replace
from pathlib import Path

import pytest

from brazil_rv.v2.round6 import (
    SESSION1_FAMILIES,
    arm_config,
    derive_session2_roster,
    families_for,
    training_command,
)


def test_round6_arms_change_only_registered_graph_components():
    names = {"slow": ["a", "b"], **{f"sidecar_{f}": ["x"] for f in SESSION1_FAMILIES}}
    parent = arm_config(names, "S0")
    for family in SESSION1_FAMILIES:
        assert arm_config(names, family) == replace(
            parent, sidecar_feature_counts=((family, 1),)
        )
    assert arm_config(names, "finetune_lr_1") == parent
    assert arm_config(names, "time_decay_756") == replace(
        parent, time_decay_half_life_sessions=756.0
    )
    assert arm_config(names, "mlp") == replace(parent, slow_encoder_kind="mlp")
    assert (
        arm_config(names, "time_decay_756", stage="P").time_decay_half_life_sessions
        is None
    )


def test_c6_excludes_zero_and_negative_points_and_resolves_ties_in_registered_order():
    points = dict(
        zip(SESSION1_FAMILIES, (0.01, 0.01, 0, -0.01, 0.005, -0.005), strict=True)
    )
    roster = derive_session2_roster(points)
    assert roster["best_single"] == "magnitudes"
    assert families_for("C6", roster) == ("magnitudes", "oddlot", "fundamentals")
    assert families_for("best_single_fresh_p", roster) == ("magnitudes",)
    assert derive_session2_roster(dict.fromkeys(SESSION1_FAMILIES, -0.001))[
        "c6_identity_s0"
    ]
    with pytest.raises(ValueError, match="six defined"):
        derive_session2_roster({**points, "lending": float("nan")})


def test_training_commands_keep_smokes_score_free_and_transfer_explicit():
    design = {
        "store": {"root": "new_store"},
        "s0_store": {"root": "parent_store"},
        "fast_initialization": {
            "mode": "native_fresh",
            "transfer_chronology_clean": True,
        },
    }
    arguments = dict(
        design=design,
        run=Path("run"),
        arm="magnitudes",
        seed=11,
        fold="F14",
        stage="F",
        checkpoint=Path("parent.pt"),
        digest="a" * 64,
        reused_s0=True,
    )
    command = training_command(**arguments)
    assert command[command.index("--pretrain-parent-store") + 1] == "parent_store"
    assert "--score-sidecar-ablations" in command and "--slow-only" in command
    smoke = training_command(**arguments, smoke=True)
    assert (
        "--score-output-dir" not in smoke and "--score-sidecar-ablations" not in smoke
    )
    assert smoke[smoke.index("--maximum-epochs") + 1] == "1"
    pretrain = training_command(design, Path("p"), "mlp", 11, "pretrain_internal", "P")
    assert (
        "--pretrain-checkpoint" not in pretrain and "--score-output-dir" not in pretrain
    )
    assert pretrain[pretrain.index("--slow-encoder-kind") + 1] == "mlp"
