import json

import numpy as np

from brazil_rv.v2 import round6_readouts as readouts


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


def test_screening_and_confirmation_use_matched_current_baselines(
    tmp_path, monkeypatch
):
    current = tmp_path / "round6"
    runs = [readouts.trajectory(current, "S0", seed, "F1") for seed in (11, 61)]
    manifest = {"selected_epoch": 2, "epochs_completed": 6}
    for run in runs:
        (run / "scores").mkdir(parents=True)
        (run / "run_manifest.json").write_text(json.dumps(manifest))
        (run / "scores/score_manifest.json").write_text("{}")
    checked, loaded = [], []

    def completed(run, *args, **kwargs):
        checked.append(run)
        return manifest

    def score_artifact(path, **kwargs):
        loaded.append((path, kwargs["expected_feature_schema_sha256"]))
        return np.arange(30).reshape(2, 3, 5), np.ones((2, 3, 5), dtype=bool)

    monkeypatch.setattr(readouts, "completed", completed)
    monkeypatch.setattr(readouts.rr, "_score_artifact", score_artifact)
    readouts.ensemble(
        {"feature_schema_sha256": "round6"}, current, "S0", "F1", (11, 61), [], []
    )
    assert checked == runs
    assert loaded == [(run / "scores", "round6") for run in runs]
