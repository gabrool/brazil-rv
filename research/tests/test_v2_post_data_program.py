import json
from dataclasses import asdict

import numpy as np
import torch

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2 import post_data_program as program
from brazil_rv.v2.round7 import configuration
from brazil_rv.v2.relational_engineering import teacher_batch


def test_matched_early_late_configuration_and_film_isolation():
    names = {
        "slow": [f"s{i}" for i in range(32)],
        "sidecar_cross_market": ["shock_oil", "exposure_oil"],
    }
    early = asdict(configuration(program.CELLS["TE_slow"], names))
    late = asdict(configuration(program.CELLS["TL_slow"], names))
    early.pop("peer_timing")
    late.pop("peer_timing")
    assert early == late
    on = configuration(program.CELLS["TE_all"], names)
    off = configuration(program.CELLS["TE_family"], names)
    assert on.family_counts == off.family_counts
    assert on.common_field_count > 0 and off.common_field_count == 0


def test_calibration_choice_uses_only_selection_and_declared_ties(tmp_path):
    for cell in program.CALIBRATION_CELLS:
        for number, recipe in enumerate(tuple(program.RECIPES)[:-1]):
            for fold in ("F2", "F14"):
                for seed in program.SEEDS[:2]:
                    write_json_atomic(
                        program.fit_path(tmp_path, cell, recipe, fold, seed)
                        / "run_manifest.json",
                        {
                            "selection_ic": [0.02, 0.0205, 0.019, 0.018, 0.017][number],
                            "selected_epoch": 3,
                            "epochs_completed": 8,
                        },
                    )
    choice = program.choose(tmp_path)
    assert set(choice["recipes"].values()) == {"sam125"}
    assert choice["evaluation_scores_read"] is False


def test_parent_plan_uses_current_pointer_and_only_authorized_calibration(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(program, "PROJECT", tmp_path)
    monkeypatch.setattr(program, "_git_identity", lambda: {"commit": "a" * 40})
    store = tmp_path / "store"
    write_json_atomic(
        store / "manifest.json",
        {
            "axes": {"date_end": "2024-12-30"},
            "feature_names": {"slow": [f"s{i}" for i in range(32)]},
            "feature_schema_sha256": "schema",
        },
    )
    write_json_atomic(
        tmp_path / "docs/v2_data_inputs.json",
        {
            "accepted": True,
            "store": {
                "root": str(store),
                "manifest_sha256": sha256_file(store / "manifest.json"),
            },
        },
    )
    registration = tmp_path / "research/preregistrations/v2_post_data.md"
    registration.parent.mkdir(parents=True)
    registration.write_text("frozen research")
    root = tmp_path / "run"
    program.freeze(root)
    result = program.plan(root, "b_parents")
    payload = json.loads((root / "b_parents_plan.json").read_text())
    assert result["jobs"] == 4
    assert {job["seed"] for job in payload["jobs"]} == {11, 29}
    assert all("--export-scores" not in job["command"] for job in payload["jobs"])
    assert all(str(store) in job["command"] for job in payload["jobs"])


def test_teacher_has_independent_dates_observed_endpoints_and_donor_only_labels():
    a, b = teacher_batch(11, "lagged"), teacher_batch(12, "lagged")
    assert not torch.equal(a["slow_features"], b["slow_features"])
    assert not a["target_mask"][:, 32:].any()
    assert not a["slow_features"][:, :32, :, 4:6].any()
    assert a["slow_features"].shape[-2] == 60
    for i in (-21, -41):
        assert a["slow_feature_mask"][:, 32:, i, 4:6][a["active_mask"][:, 32:]].all()
    assert np.isfinite(a["targets"].numpy()).all()
