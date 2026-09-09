import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from brazil_rv.v2 import round3
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.contract import TRAINING_STAGE_SCHEMA
from brazil_rv.v2.evaluate import evaluate_scores
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


def test_round3_finalizer_joins_paired_artifacts_and_preserves_comparator_masks(
    tmp_path, monkeypatch
):
    rr = round3.rr
    reference, momentum = _evaluation_pair()
    tiers = {
        "action_terms_source": "inferred_cotahist_dismes_v1",
        "schedule_source": "reconstructed_v1",
    }
    root, parent, store_root = (
        tmp_path / name for name in ("round3", "round1", "store")
    )
    for path in (root, parent, store_root):
        path.mkdir()
    for path in (
        parent / "round1_result.json",
        parent / "artifact_inventory.json",
        store_root / "manifest.json",
    ):
        write_json_atomic(path, {})
    policy_binding = {"root": str(tmp_path / "policy"), "result_sha256": "p" * 64}
    design = {
        "implementation": {"commit": "c" * 40},
        "feature_schema_sha256": "f" * 64,
        "store": {
            "root": str(store_root),
            "manifest_sha256": sha256_file(store_root / "manifest.json"),
        },
        "round1": {
            "root": str(parent),
            "result_sha256": sha256_file(parent / "round1_result.json"),
            "inventory_sha256": sha256_file(parent / "artifact_inventory.json"),
        },
        "cdi": {
            key: {"path": str(tmp_path / key), "sha256": "a" * 64}
            for key in ("development_extension", "experiment52_reference")
        },
        "bova11": {
            "root": str(tmp_path / "bova"),
            "manifest_sha256": "b" * 64,
            "data_sha256": "d" * 64,
        },
        "execution_policy": policy_binding,
        "preregistration": {"sha256": "r" * 64},
    }
    write_json_atomic(root / "frozen_design.json", design)
    dates = np.asarray(reference.inputs.dates, dtype="datetime64[D]")
    indices = {fold: np.arange(len(dates)) for fold in round3.FOLDS}
    store = SimpleNamespace(
        isins=reference.inputs.security_ids,
        manifest={"metadata": tiers},
        close=lambda: None,
    )
    monkeypatch.setattr(round3, "_design", lambda _: design)
    monkeypatch.setattr(rr, "_verify_sealed_root", lambda *a, **k: {})
    monkeypatch.setattr(rr, "_read_store_header", lambda _: (store.manifest, dates))
    monkeypatch.setattr(
        rr, "_fold_indices", lambda _: (indices, indices, indices, indices, {})
    )
    monkeypatch.setattr(rr, "_pretrain_indices", lambda _: np.arange(len(dates)))
    monkeypatch.setattr(rr, "_open_round_store", lambda *a: (store, {}))
    monkeypatch.setattr(
        rr, "_load_development_cdi", lambda **k: (np.zeros(len(dates)), {})
    )
    monkeypatch.setattr(
        rr,
        "load_bova11_series",
        lambda *a, **k: SimpleNamespace(
            manifest_sha256="b" * 64,
            data_sha256="d" * 64,
            close_by_session=np.ones(len(dates)),
        ),
    )
    monkeypatch.setattr(
        rr,
        "_load_frozen_lending",
        lambda *a, **k: SimpleNamespace(
            manifest_sha256="e" * 64, balance_sha256="f" * 64, rate_sha256="0" * 64
        ),
    )
    selected_policy = object()
    monkeypatch.setattr(
        round3, "load_selected_policy", lambda _: (selected_policy, policy_binding)
    )
    bootstrap = rr._folded_bootstrap
    monkeypatch.setattr(
        rr,
        "_folded_bootstrap",
        lambda values, **kwargs: bootstrap(values, replications=20),
    )

    for arm, seeds in (("arm_B", round3.SEEDS), ("fast_off", round3.PAIRED_SEEDS)):
        for seed in seeds:
            for fold in (
                ("pretrain_internal", *round3.FOLDS) if arm == "arm_B" else round3.FOLDS
            ):
                stage = "P" if fold == "pretrain_internal" else "F"
                run = (
                    root
                    / "trajectories"
                    / arm
                    / (
                        f"stage_P/seed_{seed}"
                        if stage == "P"
                        else f"{fold}_seed_{seed}"
                    )
                )
                run.mkdir(parents=True)
                (run / "raw_patience.pt").write_bytes(str(seed).encode())
                write_json_atomic(
                    run / "history.json",
                    [{"epoch": 1, "branch_gradient_norms": {"slow": {"mean": 1.0}}}],
                )
                write_json_atomic(
                    run / "run_manifest.json",
                    {
                        "schema": TRAINING_STAGE_SCHEMA,
                        "status": "completed",
                        **rr.RESEARCH_FLAGS,
                        **tiers,
                        "feature_schema_sha256": "f" * 64,
                        "stage": stage,
                        "seed": seed,
                        "fold": fold,
                        "model_config": {"disable_fast_stream": arm == "fast_off"},
                        "compiled_graphs": {"training": 1, "selection": 1, "total": 2},
                        "artifacts": {
                            "raw_patience.pt": sha256_file(run / "raw_patience.pt"),
                            "history.json": sha256_file(run / "history.json"),
                        },
                        "pretrain_checkpoint_sha256": sha256_file(
                            run / "raw_patience.pt"
                        )
                        if stage == "F"
                        else None,
                        "checkpoint_input_contract": {
                            "implementation_commit": "c" * 40
                        },
                        "selected_epoch": 1,
                        "epochs_completed": 1,
                    },
                )
                if stage == "F":
                    rr._persist_scores(
                        run / "scores",
                        {
                            "scores": reference.inputs.scores,
                            "score_mask": reference.inputs.score_mask,
                        },
                        tiers,
                    )
                    gate_path = run / "scores/gates.json"
                    write_json_atomic(
                        gate_path, {"archive_fast_present_counts": [1000, 1000]}
                    )
                    manifest = rr._read_json(run / "scores/score_manifest.json")
                    manifest["gate_diagnostics"] = {
                        "path": gate_path.name,
                        "sha256": sha256_file(gate_path),
                    }
                    write_json_atomic(run / "scores/score_manifest.json", manifest)
    for fold in round3.FOLDS:
        for path, value in (
            ("gbdt_ladder/b_intraday", reference),
            ("baselines/momentum_12_1", momentum),
        ):
            rr._persist_scores(
                parent / path / fold,
                {"scores": value.inputs.scores, "score_mask": value.inputs.score_mask},
                tiers,
            )

    cache, calls = {}, []

    def evaluate(**kwargs):
        assert kwargs["execution_policy"] is selected_policy
        scores, mask = kwargs["scores"], kwargs["score_mask"]
        inputs = replace(reference.inputs, scores=scores, score_mask=mask)
        key = (scores.tobytes(), mask.tobytes())
        if key not in cache:
            cache[key] = evaluate_scores(inputs, window_name="fixture")
            cache[key].report.update(rr.RESEARCH_FLAGS)
        write_json_atomic(kwargs["output"], cache[key].report)
        calls.append(kwargs["output"])
        return rr._ResearchEvaluation(result=cache[key], inputs=inputs)

    monkeypatch.setattr(rr, "_evaluate", evaluate)
    round3.finalize(root=root)
    result = rr._read_json(root / "round3_result.json")
    assert len(calls) == 24
    assert len(result["trajectory_diagnostics"]) == 27
    assert result["seed_decision"]["keep_six"] is True
    assert result["paired_deltas"]["B3_minus_fast_off"]["pooled"][
        "primary_neutral_target_ic"
    ]["estimate"] == pytest.approx(0.0)
    for fold in round3.FOLDS:
        np.testing.assert_array_equal(
            np.load(root / "aggregates/momentum" / fold / "score_mask.npy"),
            momentum.inputs.score_mask,
        )
    assert result["validation_2025_rule"]["read_is_spent"] is False
    rr.seal_root(root=root)
    inventory = rr._read_json(root / "artifact_inventory.json")
    assert inventory["status"] == "passed"
