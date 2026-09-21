import json

import numpy as np
import pytest

from brazil_rv.v2.selection_rules import (
    audit_histories,
    newey_west_mean_se,
    selection_noise_summary,
    smoothed_selection,
)


def test_window_one_is_raw_selection_and_centres_are_middle_epochs():
    scores = [0.01, 0.05, 0.02, 0.03, 0.10, 0.00]
    smoothed, centre = smoothed_selection(scores, 1)
    np.testing.assert_allclose(smoothed, scores)
    assert centre.tolist() == [0, 1, 2, 3, 4, 5]
    smoothed, centre = smoothed_selection(scores, 3)
    np.testing.assert_allclose(
        smoothed, [0.01, 0.03, 0.08 / 3, 0.10 / 3, 0.05, 0.13 / 3]
    )
    assert centre.tolist() == [0, 0, 1, 2, 3, 4]
    assert int(centre[int(smoothed.argmax())]) == 3
    with pytest.raises(ValueError):
        smoothed_selection(scores, 0)


def test_newey_west_reduces_to_iid_standard_error_without_lags():
    rng = np.random.default_rng(3)
    values = rng.normal(size=45)
    assert newey_west_mean_se(values, lags=0) == pytest.approx(
        values.std(ddof=0) / np.sqrt(45)
    )
    assert newey_west_mean_se([0.1, None], lags=10) is None
    persistent = np.repeat(rng.normal(size=9), 5)
    assert newey_west_mean_se(persistent, lags=10) > newey_west_mean_se(
        persistent, lags=0
    )


def _history(rng, means, days=45, noise=0.15):
    records = []
    best = -np.inf
    for epoch, mean in enumerate(means, start=1):
        daily = rng.normal(mean, noise, size=days)
        improved = daily.mean() > best + 1e-4
        if improved:
            best = daily.mean()
        records.append(
            {
                "epoch": epoch,
                "selection": {
                    "date_indices": list(range(days)),
                    "daily_ic": [float(x) for x in daily],
                    "mean_ic": float(daily.mean()),
                },
                "clean_fit_probe": {"mean_ic": 0.02 * epoch},
                "selected": bool(improved),
            }
        )
    return records


def test_summary_reports_noise_dominated_selection_without_evaluation_labels(tmp_path):
    rng = np.random.default_rng(11)
    history = _history(rng, [0.02, 0.03, 0.035, 0.03, 0.02, 0.01])
    summary = selection_noise_summary(history, smoothing=3, lags=10)
    assert summary["epochs"] == 6 and summary["defined_selection_days"] == 45
    assert (
        summary["selected_epoch"] == [r["epoch"] for r in history if r["selected"]][-1]
    )
    assert summary["selected_epoch_mean_se"] > 0.01
    assert summary["epochs_within_one_paired_se_of_best"] >= 1
    assert summary["selected_within_one_paired_se_of_best"] is True
    assert -1.0 <= summary["fit_selection_epoch_correlation"] <= 1.0
    fit = tmp_path / "fits" / "F2_seed_11"
    fit.mkdir(parents=True)
    (fit / "history.json").write_text(json.dumps(history))
    (fit / "run_manifest.json").write_text(json.dumps({"stage": "F"}))
    parent = tmp_path / "fits" / "P_seed_11"
    parent.mkdir()
    (parent / "history.json").write_text(
        json.dumps(_history(rng, [0.01, 0.02, 0.02], days=160))
    )
    (parent / "run_manifest.json").write_text(json.dumps({"stage": "P"}))
    audit = audit_histories(tmp_path.rglob("history.json"), stage="F")
    assert audit["pooled"]["fits"] == 1
    assert audit["rows"][0]["stage"] == "F"
    everything = audit_histories(tmp_path.rglob("history.json"))
    assert everything["pooled"]["fits"] == 2
    assert everything["pooled"]["median_selection_mean_se"] > 0
    assert (
        0.0 <= everything["pooled"]["fraction_smoothed_rule_changes_selection"] <= 1.0
    )
