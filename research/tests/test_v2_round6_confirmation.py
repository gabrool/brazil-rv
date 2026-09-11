import json

import pytest

from brazil_rv.v2 import round6_confirmation as confirmation
from brazil_rv.v2.artifacts import sha256_file


def test_confirmation_pretraining_only_for_required_graphs():
    tasks = confirmation.confirmation_tasks(
        ["S0", "magnitudes", "mlp", "C6_fresh_p"], "p"
    )
    assert len(tasks) == 9
    assert {a for a, _, _, _ in tasks} == {"S0", "mlp", "C6_fresh_p"}
    assert {s for _, s, _, _ in tasks} == {61, 79, 97}
    assert len(confirmation.confirmation_tasks(["S0", "magnitudes"], "f")) == 84


def test_confirmation_plan_uses_fresh_same_seed_P_and_cannot_launch_without_trigger(
    tmp_path, monkeypatch
):
    root = tmp_path / "run"
    root.mkdir()
    (root / "frozen_design.json").write_text("{}")
    design = {
        **confirmation.rr.DEVELOPMENT_SOURCE_TIER_LABELS,
        "store": {"root": "current_store"},
        "s0_store": {"root": "parent_store"},
        "fast_initialization": {
            "mode": "native_fresh",
            "transfer_chronology_clean": True,
        },
    }
    monkeypatch.setattr(confirmation, "design_at", lambda r: design)
    monkeypatch.setattr(
        confirmation, "session2_roster", lambda r: {"c6_families": ["magnitudes"]}
    )
    checked = []

    def completed(run, design, arm, stage, seed, fold, **kwargs):
        checked.append((arm, stage, seed, fold, run))
        return {"epochs_completed": 1, "artifacts": {"raw_patience.pt": "a" * 64}}

    monkeypatch.setattr(confirmation, "completed", completed)
    monkeypatch.setattr(
        confirmation,
        "initialization_completed",
        lambda run, design, arm, seed, **kw: completed(
            run, design, arm, "P", seed, "pretrain_internal"
        ),
    )
    decision = {
        "status": "completed",
        "frozen_design_sha256": sha256_file(root / "frozen_design.json"),
        "decision_traces": {
            "full": {
                "confirmation_arms": ["S0", "magnitudes", "mlp"],
                "confirmation_reasons": ["economics_override"],
                "confirmation_seeds": [61, 79, 97],
                "provisional_designation": "mlp",
                "economics_override": "mlp",
                "ic_leader": "magnitudes",
            }
        },
    }
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(decision))
    confirmation.write_plan(root, path, "f")
    plan = json.loads((root / "round6_plan_confirmation_f.json").read_text())
    assert len(plan["jobs"]) == 126
    for job in plan["jobs"]:
        cmd = job["command"]
        own = "mlp" if job["name"].startswith("mlp_") else "S0"
        checkpoint = (
            root
            / "trajectories"
            / own
            / "stage_P"
            / f"seed_{job['seed']}"
            / "raw_patience.pt"
        )
        assert cmd[cmd.index("--pretrain-checkpoint") + 1] == str(checkpoint)
        if own == "S0":
            assert cmd[cmd.index("--pretrain-parent-store") + 1] == "parent_store"
        else:
            assert "--pretrain-parent-store" not in cmd
        assert cmd[cmd.index("--maximum-epochs") + 1] == "60"
        assert job["expected_manifest"]["scoring_complete"]
    decision["decision_traces"]["full"]["confirmation_reasons"] = []
    path.write_text(json.dumps(decision))
    with pytest.raises(ValueError, match="did not trigger"):
        confirmation.write_plan(root, path, "p")


def test_confirmation_parent_keeps_original_precision_sampling_and_budget(tmp_path):
    command = [
        "python",
        "-m",
        "brazil_rv.v2.train",
        "--maximum-epochs",
        "60",
        "--selection-interval",
        "2",
        "--use-bf16",
        "--slow-only",
    ]
    original = confirmation.initialization_command(command, tmp_path, smoke=False)
    assert original[original.index("--maximum-epochs") + 1] == "20"
    assert "--use-bf16" not in original and "--selection-interval" not in original
    assert repr(str(tmp_path / "research/src")) in original[2]
    assert confirmation.confirmation_tasks(["S0", "mlp", "magnitudes"], "smoke") == [
        ("S0", 11, "pretrain_internal", "P"),
        ("mlp", 11, "pretrain_internal", "P"),
    ]
