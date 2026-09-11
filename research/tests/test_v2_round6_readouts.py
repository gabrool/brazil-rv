import json

import numpy as np
import pytest

from brazil_rv.v2 import round6_readouts as readouts
from brazil_rv.v2.artifacts import sha256_file


def test_attribution_requires_identical_seed_roster_and_trained_checkpoints():
    records = [{"run_manifest_sha256": "model11"}, {"run_manifest_sha256": "model29"}]
    metadata = {"arm": "magnitudes", "seeds": [11, 29], "trajectories": records}
    assert readouts.attribution_reference_matches(
        metadata, records, (11, 29), "magnitudes"
    )
    assert not readouts.attribution_reference_matches(
        metadata, records, (29, 47), "magnitudes"
    )
    assert not readouts.attribution_reference_matches(
        metadata, records, (11, 29), "events"
    )
    changed = [{"run_manifest_sha256": "another_fit"}, records[1]]
    assert not readouts.attribution_reference_matches(
        metadata, changed, (11, 29), "magnitudes"
    )


def test_recovered_parent_hashes_and_fresh_confirmation_are_separate(
    tmp_path, monkeypatch
):
    parent = tmp_path / "sealed_parent"
    current = tmp_path / "round6"
    store = tmp_path / "parent_store"
    store.mkdir()
    (store / "manifest.json").write_text('{"feature_schema_sha256": "parent"}')
    original = readouts.trajectory(parent, "S0", 11, "F1")
    fresh = readouts.trajectory(current, "S0", 61, "F1")
    manifest = {"selected_epoch": 1, "epochs_completed": 2}
    files = []
    for run in (original, fresh):
        (run / "scores").mkdir(parents=True)
        (run / "run_manifest.json").write_text(json.dumps(manifest))
        for name in (
            "score_manifest.json",
            "scores.npy",
            "score_mask.npy",
            "date_index.npy",
            "isin_index.npy",
        ):
            (run / "scores" / name).write_text("{}")
        if run == original:
            for path in sorted(run.rglob("*")):
                if path.is_file():
                    files.append(
                        {
                            "path": path.relative_to(parent).as_posix(),
                            "sha256": sha256_file(path),
                        }
                    )
    inventory = parent / "artifact_inventory.json"
    inventory.write_text(json.dumps({"files": files}))
    design = {
        "s0_panels": {
            "root": "/remote/sealed_parent",
            "inventory_sha256": sha256_file(inventory),
        },
        "s0_store": {"root": "/remote/parent_store"},
        "feature_schema_sha256": "round6",
    }
    monkeypatch.setattr(
        readouts,
        "resolve_external_root",
        lambda path: (
            {"/remote/sealed_parent": parent, "/remote/parent_store": store}[path],
            None,
        ),
    )
    monkeypatch.setattr(
        readouts.rr, "_assert_current_clean_training", lambda *a, **k: None
    )
    checked = []

    def completed(run, *args, **kwargs):
        checked.append(run)
        return manifest

    monkeypatch.setattr(readouts, "completed", completed)
    loaded = []

    def score_artifact(path, **kwargs):
        loaded.append((path, kwargs["expected_feature_schema_sha256"]))
        return np.arange(30).reshape(2, 3, 5), np.ones((2, 3, 5), dtype=bool)

    monkeypatch.setattr(readouts.rr, "_score_artifact", score_artifact)
    readouts.ensemble(design, current, "S0", "F1", (11, 61), [], [])
    assert loaded == [(original / "scores", "parent"), (fresh / "scores", "round6")]
    assert checked == [fresh]
    assert design["s0_panels"]["root"] == "/remote/sealed_parent"

    (original / "scores/scores.npy").write_text("changed")
    with pytest.raises(ValueError, match="sealed parent artifact changed"):
        readouts.ensemble(design, current, "S0", "F1", (11,), [], [])
