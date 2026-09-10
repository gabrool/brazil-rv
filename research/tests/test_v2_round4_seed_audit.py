from brazil_rv.v2.round4_seed_audit import development_decision
import pytest


@pytest.mark.parametrize("net_lower,expected", [(-0.1, "S0"), (0.1, "fast_off")])
def test_pooled_pair_adapter_preserves_selection_and_excludes_failed_arm(
    net_lower, expected
):
    from brazil_rv.v2.round4_seed_audit import _trace

    ic = "primary_neutral_target_ic"
    net = "headline_net_excess_bps"
    readouts = {
        "fast_off": {ic: {"estimate": 0.02}, net: {"estimate": 5.0}},
        "S0": {ic: {"estimate": 0.03}, net: {"estimate": 4.0}},
        "P": {ic: {"estimate": 0.04}, net: {"estimate": 6.0}},
    }
    comparisons = {
        "fast_off_minus_S0": {
            "pooled": {
                ic: {"lower_95": -0.02, "upper_95": 0.01},
                net: {"lower_95": net_lower, "upper_95": 2.0},
            }
        },
        "S0_minus_fast_off": {
            "pooled": {},
            "informative_subsets": {
                "S0": {"pooled": {ic: {"lower_95": -0.01, "upper_95": 0.02}}}
            },
        },
    }
    trace = _trace(readouts, comparisons, excluded={"P"})
    assert trace["ic_leader"] == "S0"
    assert (trace["economics_override"] or trace["ic_leader"]) == expected
    assert trace["provisional_parent"] == "S0"
    assert "P" not in trace["eligible"]
    assert trace["designation"] is None


def test_only_isolated_nonbaseline_occupancy_failure_rejects_one_arm():
    from brazil_rv.v2.round4_seed_audit import isolated_occupancy_failure

    flag = "mean_quintile_occupancy_deviation_long_above_two"
    occupancy = RuntimeError(f"registered book stop: {{'borrow_balance': ['{flag}']}}")
    assert isolated_occupancy_failure(occupancy, "P")
    assert not isolated_occupancy_failure(occupancy, "fast_off")
    mixed = RuntimeError(
        f"registered book stop: {{'borrow_balance': ['{flag}', 'D4_cap_block_defects']}}"
    )
    assert not isolated_occupancy_failure(mixed, "P")
    assert not isolated_occupancy_failure(
        RuntimeError("network identity mismatch"), "P"
    )


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


def test_audit_loads_canonical_network_artifacts_with_frozen_axes(
    tmp_path, monkeypatch
):
    import json
    from types import SimpleNamespace

    import numpy as np
    import pytest

    from brazil_rv.v2 import round4_seed_audit as audit
    from brazil_rv.v2.artifacts import sha256_file

    dates = np.asarray(["2024-01-02", "2024-01-03"], dtype="datetime64[D]")
    isins = ("BR1", "BR2", "BR3")
    tiers = {
        "action_terms_source": "inferred_cotahist_dismes_v1",
        "schedule_source": "reconstructed_v1",
    }
    source = tmp_path / "source"
    source.mkdir()
    (source / "frozen_design.json").write_text(
        json.dumps(
            {
                "store": {"root": "unused"},
                "feature_schema_sha256": "c" * 64,
                "execution_policy": {"root": "unused", "result_sha256": "d" * 64},
            }
        )
    )
    baseline = source / "cpu_replay/baselines/momentum_12_1/F1"
    baseline.mkdir(parents=True)
    (baseline / "evaluation.json").write_text('{"source_artifact_hashes": {}}')
    for seed in (11, 29, 47):
        run = source / "trajectories/fast_off" / f"F1_seed_{seed}"
        scores = run / "scores"
        scores.mkdir(parents=True)
        (run / "run_manifest.json").write_text("{}")
        arrays = {
            "scores.npy": np.zeros((2, 3, 5), dtype=np.float32),
            "score_mask.npy": np.ones((2, 3, 5), dtype=bool),
            "date_index.npy": dates,
            "isin_index.npy": np.asarray(isins),
        }
        records = {}
        for name, values in arrays.items():
            path = scores / name
            np.save(path, values, allow_pickle=False)
            records[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        (scores / "score_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "BRAZIL_RV_V2_SCORE_ARTIFACT_V2",
                    "status": "completed",
                    **audit.rr.RESEARCH_FLAGS,
                    **tiers,
                    "feature_schema_sha256": "c" * 64,
                    "artifacts": records,
                }
            )
        )
    closed = []
    context = SimpleNamespace(
        store=SimpleNamespace(
            isins=isins, manifest={}, close=lambda: closed.append(True)
        ),
        evaluation={"F1": np.arange(2)},
    )
    monkeypatch.setattr(audit, "DEVELOPMENT_FOLDS", ("F1",))
    monkeypatch.setattr(audit, "ARMS", ("fast_off",))
    monkeypatch.setattr(audit.rr, "_read_store_header", lambda _: ({}, dates))
    monkeypatch.setattr(audit.rr, "_open_ledger_replay", lambda _: context)
    monkeypatch.setattr(audit.rr, "load_selected_policy", lambda *a, **k: (None, None))
    monkeypatch.setattr(audit.rr, "_source_tier_labels", lambda _: tiers)

    class LoadedCanonicalScores(Exception):
        pass

    def capture(path, arrays, metadata):
        assert metadata["seeds"] == [29, 47]
        assert arrays["scores"].shape == (2, 3, 5)
        raise LoadedCanonicalScores

    monkeypatch.setattr(audit.rr, "_persist_scores", capture)
    with pytest.raises(LoadedCanonicalScores):
        audit._audit_omission(source, tmp_path / "output", 11)
    assert closed == [True]
