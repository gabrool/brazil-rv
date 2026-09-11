from copy import deepcopy
import json
from types import SimpleNamespace

from brazil_rv.v2.round4_seed_audit import isolated_occupancy_failure
from brazil_rv.v2.round6_decisions import IC, NET, promotion_trace, seed_stability
from brazil_rv.v2 import round6_decisions as decisions
from brazil_rv.v2 import round6_seed_audit as seed_audit
from brazil_rv.v2.artifacts import sha256_file


def interval(point, low=None, high=None):
    return {"estimate": point, "lower_95": low, "upper_95": high}


def pair(arm, ic, net):
    return {
        "pooled": {IC: ic, NET: net},
        "informative_subsets": {
            arm: {"folds": ["F6", "F7"], "pooled": {IC: ic, NET: net}}
        },
    }


def test_economics_override_confirms_both_claims_and_excludes_unusable_arms():
    rows = {
        "S0": {IC: interval(0.02), NET: interval(4)},
        "magnitudes": {IC: interval(0.03), NET: interval(5)},
        "lending": {IC: interval(0.025), NET: interval(9)},
        "undefined": {IC: interval(0.8), NET: interval(None)},
        "negative": {IC: interval(0.9), NET: interval(-1)},
        "failed_gate": {IC: interval(0.99), NET: interval(50)},
    }
    pairs = {
        "S0_minus_magnitudes": pair(
            "S0", interval(-0.01, -0.02, -0.001), interval(-1, -2, 0)
        ),
        "lending_minus_magnitudes": pair(
            "lending", interval(-0.005, -0.02, 0.01), interval(4, 1, 7)
        ),
        "lending_minus_S0": pair(
            "lending", interval(0.005, 0.002, 0.01), interval(5, 1, 9)
        ),
    }
    trace = promotion_trace(rows, pairs, excluded=("failed_gate",))
    assert trace["eligible"] == ["S0", "magnitudes", "lending"]
    assert trace["ic_leader"] == "magnitudes"
    assert trace["provisional_designation"] == "lending"
    assert trace["confirmation_arms"] == ["S0", "magnitudes", "lending"]
    assert trace["confirmation_reasons"] == ["economics_override"]


def test_confirmation_uses_informative_lower_bound_including_negative_boundary():
    rows = {
        "S0": {IC: interval(0.02), NET: interval(4)},
        "lending": {IC: interval(0.025), NET: interval(4)},
    }
    pairs = {
        "S0_minus_lending": pair(
            "S0", interval(-0.005, -0.02, 0.01), interval(0, -1, 1)
        ),
        "lending_minus_S0": pair(
            "lending", interval(0.005, 0.002, 0.01), interval(0, -1, 1)
        ),
    }
    for lower in (-0.001, 0, 0.001):
        pairs["lending_minus_S0"]["informative_subsets"]["lending"]["pooled"] = {
            IC: interval(0.005, lower, 0.01),
            NET: interval(0, -1, 1),
        }
        trace = promotion_trace(rows, pairs)
        assert trace["confirmation_arms"] == ["S0", "lending"]
    pairs["lending_minus_S0"]["informative_subsets"]["lending"]["pooled"][IC][
        "lower_95"
    ] = 0.001001
    assert promotion_trace(rows, pairs)["confirmation_arms"] == []
    # Retaining the parent cannot trigger a meaningless self-comparison refit.
    assert promotion_trace({"S0": rows["S0"]}, {})["confirmation_arms"] == []


def test_all_fixed_omissions_must_agree_without_selecting_a_favorable_subset():
    trace = {"provisional_designation": "lending", "eligible": ["S0", "lending"]}
    panels = {k: deepcopy(trace) for k in ("full", "omit_11", "omit_29", "omit_47")}
    assert seed_stability(panels)["research_designation"] == "lending"
    panels["omit_29"]["provisional_designation"] = "S0"
    decision = seed_stability(panels)
    assert decision["research_designation"] is None
    assert decision["working_research_comparator"] == "S0"
    assert decision["parent_inconclusive"]


def test_only_isolated_nonbaseline_occupancy_can_be_rejected_locally():
    occupancy = RuntimeError(
        "registered book stop: {'headline': ['mean_quintile_occupancy_deviation_long_above_two']}"
    )
    assert isolated_occupancy_failure(occupancy, "lending", baseline="S0")
    assert not isolated_occupancy_failure(occupancy, "S0", baseline="S0")
    mixed = RuntimeError(
        "registered book stop: {'headline': ['mean_quintile_occupancy_deviation_long_above_two', 'd1_failure']}"
    )
    assert not isolated_occupancy_failure(mixed, "lending", baseline="S0")
    assert not isolated_occupancy_failure(
        RuntimeError("identity mismatch"), "lending", baseline="S0"
    )


def test_era_readout_preserves_calendar_missingness_and_fold_boundaries(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(decisions, "DEVELOPMENT_FOLDS", ("F1", "F2"))
    for fold, year, value in (("F1", 2018, 0.01), ("F2", 2020, -0.02)):
        rows = [
            {"date": f"{year}-01-{day:02d}", "delta": value if day <= 5 else None}
            for day in range(1, 21)
        ]
        (tmp_path / f"{fold}.json").write_text(
            json.dumps({"population_audit": {fold: {IC: rows}}})
        )
    result = decisions.era_readout(tmp_path)
    assert set(result["eras"]) == {"2018_2019", "2020_2021"}
    row = result["eras"]["2020_2021"]["paired"][IC]
    assert row["estimate"] == -0.02
    assert row["possible_observations"] == 20
    assert row["finite_observations"] == 5
    assert row["fold_boundary_preserved"]


def test_finished_seed_audit_binds_the_design_consumed_by_confirmation(
    tmp_path, monkeypatch
):
    root, review, output = (tmp_path / p for p in ("root", "review", "audit"))
    root.mkdir()
    review.mkdir()
    (root / "frozen_design.json").write_text('{"input_coverage": {"folds": {}}}')
    digest = sha256_file(root / "frozen_design.json")
    rows = {"S0": {IC: interval(0.02), NET: interval(4)}}
    full = {
        "frozen_design_sha256": digest,
        "readouts": rows,
        "registered_C6_roster": {"c6_families": []},
        "paired": {},
    }
    (review / "result.json").write_text(json.dumps(full))
    for seed in (11, 29, 47):
        dest = output / f"omit_{seed}"
        dest.mkdir(parents=True)
        (dest / "result.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "review_sha256": sha256_file(review / "result.json"),
                    "seeds": [s for s in (11, 29, 47) if s != seed],
                    "readouts": rows,
                    "paired": {},
                    "rejected_books": {},
                    "C6_roster_sensitivity": {},
                }
            )
        )
    monkeypatch.setattr(seed_audit, "evaluation_design", lambda d: d)
    monkeypatch.setattr(
        seed_audit.rr,
        "_open_ledger_replay",
        lambda d: SimpleNamespace(store=SimpleNamespace(close=lambda: None)),
    )
    seed_audit.finish(root, review, output)
    result = json.loads((output / "seed_audit_result.json").read_text())
    assert result["frozen_design_sha256"] == digest
    assert result["decision"]["research_designation"] == "S0"
    assert result["decision_traces"]["full"]["confirmation_arms"] == []
