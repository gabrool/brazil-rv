import json

import pytest
import numpy as np
from types import SimpleNamespace

from brazil_rv.v2 import round4
from brazil_rv.v2.contract import DEVELOPMENT_FOLDS
from brazil_rv.v2.train import _train_parser


def test_informative_subsets_read_only_validity_and_lending_excludes_decision_row(
    tmp_path, monkeypatch
):
    days = np.arange(np.datetime64("2018-01-01"), np.datetime64("2018-04-21"))
    arrays = {
        "active": np.ones((110, 2), bool),
        "slow_timestep_valid": np.ones((110, 2), bool),
        "intraday_valid": np.zeros((110, 2, 1), bool),
        "sidecar_lending_valid": np.zeros((110, 2, 1), bool),
    }
    arrays["intraday_valid"][84, 0, 0] = True
    arrays["sidecar_lending_valid"][84, 0, 0] = True
    store = SimpleNamespace(
        read=lambda key, rows: arrays[key][rows], close=lambda: None
    )
    windows = {"F1": np.arange(60, 85), "F2": np.arange(85, 110)}
    monkeypatch.setattr(round4, "DEVELOPMENT_FOLDS", ("F1", "F2"))
    monkeypatch.setattr(round4.rr, "_read_store_header", lambda _: ({}, days))
    monkeypatch.setattr(
        round4.rr, "_fold_indices", lambda _: (None, None, windows, None, None)
    )
    monkeypatch.setattr(
        round4.rr,
        "open_store_for_samples",
        lambda *a, **k: (store, SimpleNamespace(payload=lambda: {})),
    )
    (tmp_path / "manifest.json").write_text("{}")
    result = round4.informative_fold_protocol(tmp_path)
    assert result["folds"]["S0"] == ["F1"]
    assert result["folds"]["L"] == ["F2"]
    assert result["coverage"]["L"]["F1"]["informative_sessions"] == 0
    arrays["intraday_valid"][109] = True
    assert (
        round4.informative_fold_protocol(tmp_path)["coverage"]["S0"]["F1"]
        == result["coverage"]["S0"]["F1"]
    )


def test_paired_informative_subsets_keep_whole_folds_and_reverse_intervals(
    tmp_path, monkeypatch
):
    from brazil_rv.v2 import checkpoint_readouts as cr

    monkeypatch.setattr(cr, "retained", lambda context, path, fold: None)

    def paired(left, right):
        fold = next(iter(left))
        value = 0.0 if fold == "F1" else 1.0
        return {
            "population_audit": {
                fold: {
                    "primary_neutral_target_ic": [{"delta": value} for _ in range(25)]
                }
            },
            "folds": {fold: {"primary_neutral_target_ic": {"estimate": value}}},
        }

    monkeypatch.setattr(cr.rr, "_paired_readouts", paired)
    paths = {
        name: {fold: tmp_path for fold in ("F1", "F2")} for name in ("L", "fast_off")
    }
    result = cr.paired_readouts(
        None,
        paths,
        tmp_path / "pairs",
        informative_folds={"L": ["F2"], "fast_off": ["F1", "F2"]},
    )
    forward = result["L_minus_fast_off"]
    assert forward["pooled"]["primary_neutral_target_ic"]["estimate"] == 0.5
    subset = forward["informative_subsets"]["L"]
    assert subset["folds"] == ["F2"]
    assert subset["pooled"]["primary_neutral_target_ic"]["estimate"] == 1.0
    reverse = result["fast_off_minus_L"]["informative_subsets"]["L"]["pooled"][
        "primary_neutral_target_ic"
    ]
    assert reverse["lower_95"] == reverse["upper_95"] == -1.0


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
    monkeypatch.setattr(
        round4, "required_confirmation_arms", lambda _: ("fast_off", "S0", "H")
    )
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
    subset = {"folds": ["F8", "F9"], "pooled": comparisons["S0_minus_fast_off"]}
    screened = round4.promotion_trace(
        readouts, comparisons, confirmed=False, s0_informative=subset
    )
    assert screened["ic_leader"] == "fast_off" and "H" not in screened["eligible"]
    assert screened["economics_override"] == "S0"
    assert screened["designation"] is screened["next_round_parent"] is None
    confirmed = round4.promotion_trace(
        readouts, comparisons, confirmed=True, s0_informative=subset
    )
    assert confirmed["designation"] == confirmed["next_round_parent"] == "S0"
    comparisons["S0_minus_fast_off"]["primary_neutral_target_ic"] = interval(
        -0.002, -0.003, -0.001
    )
    assert (
        round4.promotion_trace(
            readouts, comparisons, confirmed=True, s0_informative=subset
        )["next_round_parent"]
        == "fast_off"
    )
    readouts["S0"]["headline_net_excess_bps"]["estimate"] = -1
    assert (
        round4.promotion_trace(
            readouts, comparisons, confirmed=True, s0_informative=subset
        )["designation"]
        == "fast_off"
    )


def test_s0_tie_uses_informative_subset_even_if_all_folds_look_better():
    levels = {
        a: {
            "primary_neutral_target_ic": {"estimate": 0.03},
            "headline_net_excess_bps": {"estimate": 1.0},
        }
        for a in ("fast_off", "S0")
    }
    comparisons = {
        "S0_minus_fast_off": {
            "primary_neutral_target_ic": {
                "estimate": 0.001,
                "lower_95": -0.001,
                "upper_95": 0.003,
            },
            "headline_net_excess_bps": {"estimate": 0, "lower_95": -1, "upper_95": 1},
        }
    }
    subset = {
        "folds": ["F8"],
        "pooled": {
            "primary_neutral_target_ic": {
                "estimate": -0.003,
                "lower_95": -0.005,
                "upper_95": -0.001,
            }
        },
    }
    trace = round4.promotion_trace(
        levels, comparisons, confirmed=True, s0_informative=subset
    )
    assert trace["next_round_parent"] == "fast_off"
    assert trace["S0_informative_comparison"] == subset


def test_confirmation_includes_every_override_qualifier_and_mandatory_s0():
    assert round4.required_confirmation_arms(
        {
            "promotion_trace": {
                "ic_leader": "H",
                "economics_override_qualifiers": ["P", "L"],
            }
        }
    ) == ("fast_off", "S0", "H", "P", "L")
