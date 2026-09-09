import json

import pytest

from brazil_rv.v2 import round3
from test_v2_research_rounds import _evaluation_pair


def test_round3_plans_preserve_seed_pairs_and_ablation_scope(tmp_path, monkeypatch):
    design = {
        "store": {"root": str(tmp_path / "store")},
        "enabled_sidecars": [],
        "fast_initialization": {
            "mode": "native_fresh",
            "transfer_chronology_clean": True,
        },
        "max_parallel_trajectories": 6,
        "action_terms_source": "inferred_cotahist_dismes_v1",
        "schedule_source": "reconstructed_v1",
    }
    monkeypatch.setattr(round3, "_design", lambda _: design)

    def completed(run, **identity):
        return {
            "epochs_completed": 1,
            "artifacts": {"raw_patience.pt": str(identity["seed"])},
        }

    monkeypatch.setattr(round3, "_completed", completed)
    for phase in ("smoke", "p", "main"):
        round3.write_plan(root=tmp_path, phase=phase)
    plans = {
        phase: json.loads((tmp_path / f"round3_plan_{phase}.json").read_text())
        for phase in ("smoke", "p", "main")
    }
    assert [len(plans[phase]["jobs"]) for phase in plans] == [2, 6, 27]
    assert plans["smoke"]["max_parallel"] == 1
    assert all(
        "--score-output-dir" not in job["command"] for job in plans["smoke"]["jobs"]
    )
    for job in plans["main"]["jobs"]:
        command = job["command"]
        disabled = job["name"].startswith("fast_off")
        assert ("--disable-fast-stream" in command) == disabled
        assert not disabled or job["seed"] in (11, 29, 47)
        assert command[command.index("--pretrain-sha256") + 1] == str(job["seed"])
        assert "--record-branch-diagnostics" in command
        assert command[command.index("--maximum-epochs") + 1] == "20"


def test_round3_per_horizon_retains_exact_primary_common_population():
    candidate, baseline = _evaluation_pair()
    paired = round3.paired_horizons(
        {fold: candidate for fold in round3.FOLDS},
        {fold: baseline for fold in round3.FOLDS},
    )
    for horizon in (1, 2, 3, 5):
        value = paired[f"D{horizon}"]
        assert value["pooled"]["estimate"] == pytest.approx(2.0)
        assert value["pooled"]["finite_observations"] == 60
        assert (
            value["population_audit"]["F1"][0]["common_candidate_baseline_name_count"]
            == 70
        )
    assert paired["D10"]["pooled"]["finite_observations"] == 45


@pytest.mark.parametrize(
    "lower,expected", [(0.015, True), (0.01499, False), (None, False)]
)
def test_validation_rule_is_inclusive_and_never_opens_holdout(lower, expected):
    report = round3.validation_readout(
        {
            "pooled": {
                "primary_neutral_target_ic": {"lower_95": lower},
                "headline_net_excess_bps": {"estimate": 3.0, "lower_95": -2.0},
            }
        },
        {"estimate": -5.0},
    )
    assert report["development_rule_met"] is expected
    assert report["read_is_spent"] is False
    assert report["official_validation_opened"] is False
