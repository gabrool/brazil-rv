from brazil_rv.v2.round4_seed_audit import development_decision


def panel(leader="H", parent="S0", eligible=("fast_off", "S0", "H")):
    return {
        "ic_leader": leader,
        "economics_override": None,
        "provisional_parent": parent,
        "eligible": list(eligible),
    }


def test_one_omission_disagreement_cannot_be_hidden_by_two_matching_panels():
    result = development_decision(
        panel(), {11: panel(), 29: panel(), 47: panel("P", "fast_off")}
    )
    assert result["research_designation"] is None
    assert result["working_research_parent"] == "fast_off"
    assert result["parent_inconclusive"]


def test_stable_research_decisions_do_not_claim_confirmation_or_holdout_access():
    result = development_decision(panel(), {seed: panel() for seed in (11, 29, 47)})
    assert result["research_designation"] == "H"
    assert result["working_research_parent"] == "S0"
    assert not result["confirmed_six_seed_panel"]
    assert not result["read_2025_authorized"]
    assert not result["independent_replication"]


def test_matching_but_ineligible_parent_is_not_accepted():
    result = development_decision(
        panel(parent="fast_off"),
        {
            11: panel(parent="fast_off"),
            29: panel(parent="fast_off"),
            47: panel(parent="fast_off", eligible=("H",)),
        },
    )
    assert not result["parent_stable"]
    assert result["parent_inconclusive"]
