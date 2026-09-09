import json

import pytest

from brazil_rv.v2 import round4
from brazil_rv.v2.contract import DEVELOPMENT_FOLDS
from brazil_rv.v2.train import _train_parser


def _design():
    return {
        "store": {"root": "store"},
        "enabled_sidecars": [],
        "fast_initialization": {
            "mode": "native_fresh",
            "transfer_chronology_clean": True,
        },
        "action_terms_source": "inferred_cotahist_dismes_v1",
        "schedule_source": "reconstructed_v1",
    }


def test_freeze_rejects_a_cpu_root_that_no_longer_matches_its_seal(
    tmp_path, monkeypatch
):
    from brazil_rv.v2.artifacts import write_json_atomic

    monkeypatch.setattr(round4.rr, "_git_identity", lambda: {})
    write_json_atomic(
        tmp_path / "checkpoint_cpu_result.json",
        {
            "schema": round4.CHECKPOINT_SCHEMA,
            "status": "cpu_rebaseline_complete",
        },
    )
    write_json_atomic(
        tmp_path / "artifact_inventory.json",
        {
            "status": "passed",
            "files": [],
            "excluded_self": [
                "artifact_inventory.json",
                "artifact_inventory.json.sha256",
            ],
        },
    )
    with pytest.raises(ValueError, match="verified sealed CPU root"):
        round4.freeze(tmp_path, tmp_path / "round4")


def test_phase_counts_same_seed_handoffs_and_cli_options(tmp_path, monkeypatch):
    monkeypatch.setattr(round4, "_design", lambda root: _design())
    monkeypatch.setattr(
        round4,
        "completed",
        lambda *a, **k: {
            "epochs_completed": 1,
            "artifacts": {"raw_patience.pt": "a" * 64},
        },
    )
    monkeypatch.setattr(round4, "cpu_cell_completed", lambda root: True)
    (tmp_path / "screening_result.json").write_text("{}")
    counts = {
        "smoke": 6,
        "p": 12,
        "parent": 42,
        "arms": 210,
        "selection": 9,
        "confirmation_p": 6,
        "confirmation": 126,
    }
    for phase, count in counts.items():
        round4.write_plan(tmp_path, phase, confirmation_arms=("H",))
        plan = json.loads((tmp_path / f"round4_plan_{phase}.json").read_text())
        assert len(plan["jobs"]) == count
        assert plan["max_parallel"] == (1 if phase == "smoke" else 6)
        assert plan["first_failure_stop"] is True
        for job in plan["jobs"]:
            args = _train_parser().parse_args(job["command"][3:])
            assert args.disable_fast_stream
            if args.stage == "F" and phase != "smoke":
                graph = round4.parent_graph(job["name"].split("_F_")[0])
                assert (
                    args.pretrain_checkpoint
                    == tmp_path
                    / "trajectories"
                    / graph
                    / "stage_P"
                    / f"seed_{args.seed}"
                    / "raw_patience.pt"
                )
                assert args.pretrain_sha256 == "a" * 64
            if job["name"].startswith("H_"):
                assert tuple(args.horizon_loss_weights) == round4.WEIGHTS_H
            if job["name"].startswith("P_"):
                assert args.lambda_persistence == 0.1
            if phase == "selection":
                assert tuple(args.selection_horizons) == (1, 2, 3, 5)
                assert args.fold in DEVELOPMENT_FOLDS[-3:]
            if phase == "smoke":
                assert args.maximum_epochs == 1 and args.score_output_dir is None
            if phase.startswith("confirmation"):
                assert args.seed in (61, 79, 97)


def test_arms_require_parent_acceptance_and_confirmation_requires_screen(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(round4, "_design", lambda root: _design())
    monkeypatch.setattr(round4, "completed", lambda *a, **k: {"epochs_completed": 1})
    monkeypatch.setattr(round4, "cpu_cell_completed", lambda root: False)
    with pytest.raises(ValueError, match="accept the re-baselined parent"):
        round4.write_plan(tmp_path, "arms")
    monkeypatch.setattr(round4, "cpu_cell_completed", lambda root: True)
    with pytest.raises(ValueError, match="complete screening readout"):
        round4.write_plan(tmp_path, "confirmation_p", confirmation_arms=("S0",))


def test_promotion_uses_paired_intervals_eligibility_and_confirmation():
    def interval(point, low, high):
        return {"estimate": point, "lower_95": low, "upper_95": high}

    readouts = {
        arm: {
            "primary_neutral_target_ic": {"estimate": ic},
            "headline_net_excess_bps": {"estimate": net},
        }
        for arm, ic, net in (("fast_off", 0.03, 1), ("S0", 0.029, 1.5), ("H", 0.04, -1))
    }
    comparisons = {
        "S0_minus_fast_off": {
            "primary_neutral_target_ic": interval(-0.001, -0.002, 0.001),
            "headline_net_excess_bps": interval(0.5, 0.1, 0.9),
        }
    }
    screened = round4.promotion_trace(readouts, comparisons, confirmed=False)
    assert screened["ic_leader"] == "fast_off" and "H" not in screened["eligible"]
    assert screened["economics_override"] == "S0"
    assert screened["designation"] is screened["next_round_parent"] is None
    confirmed = round4.promotion_trace(readouts, comparisons, confirmed=True)
    assert confirmed["designation"] == confirmed["next_round_parent"] == "S0"
    comparisons["S0_minus_fast_off"]["primary_neutral_target_ic"] = interval(
        -0.002, -0.003, -0.001
    )
    assert (
        round4.promotion_trace(readouts, comparisons, confirmed=True)[
            "next_round_parent"
        ]
        == "fast_off"
    )
    readouts["S0"]["headline_net_excess_bps"]["estimate"] = -1
    assert (
        round4.promotion_trace(readouts, comparisons, confirmed=True)["designation"]
        == "fast_off"
    )
