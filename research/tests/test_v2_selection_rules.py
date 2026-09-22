import json

import numpy as np
import pytest

from brazil_rv.v2.selection_rules import (
    audit_histories,
    epoch_views,
    newey_west_mean_se,
    raw_prefix_view,
    raw_selected_epoch,
    selection_noise_summary,
    smoothed_selection,
    smoothed_view,
)


def _trajectory(scores, patience=None, minimum_improvement=1e-4):
    """Records as the trainer writes them, with its own improvement flags."""
    records, best = [], -np.inf
    for epoch, score in enumerate(scores, start=1):
        improved = score > best + minimum_improvement
        if improved:
            best = score
        records.append(
            {
                "epoch": epoch,
                "selection": {"mean_ic": score, "daily_ic": [score] * 45},
                "selected": improved,
            }
        )
    return records


def test_raw_prefix_view_reproduces_the_matched_stopping_selector():
    history = _trajectory([0.01, 0.05, 0.02, 0.03, 0.04, 0.045, 0.049, 0.10, 0.0])
    five = raw_prefix_view(history, patience=5, minimum_improvement=1e-4)
    assert five["selected_epoch"] == 2 and five["epochs_considered"] == 7
    assert five["stop_reason"] == "patience"
    twenty = raw_prefix_view(history, patience=20, minimum_improvement=1e-4)
    assert twenty["selected_epoch"] == 8 and twenty["stop_reason"] == "trajectory_end"
    assert raw_selected_epoch(history) == 8


def test_smoothed_view_selects_the_centre_of_the_best_window():
    history = _trajectory([0.01, 0.05, 0.02, 0.03, 0.10, 0.00, 0.00, 0.00])
    view = smoothed_view(history, window=3)
    assert view["selected_epoch"] == 4 and view["stop_reason"] == "trajectory_end"
    assert view["selection_score"] == pytest.approx(np.mean([0.02, 0.03, 0.10]))
    prefix = smoothed_view(history, window=3, patience=2)
    # Smoothed scores: .01,.03,.0267,.0333,.05,... the .0267 dip costs one stale
    # epoch, epoch 4 recovers, epoch 5 improves, then two stale epochs stop it.
    assert prefix["selected_epoch"] == 4 and prefix["epochs_considered"] == 7
    assert prefix["stop_reason"] == "patience"
    with pytest.raises(ValueError):
        smoothed_view([{"epoch": 2, "selection": {"mean_ic": 0.1}}], window=3)


def test_epoch_views_name_only_saved_epochs():
    history = _trajectory([0.01, 0.05, 0.02, 0.03, 0.10, 0.00, 0.00, 0.00])
    views = epoch_views(history)
    assert views["raw"]["epochs"] == [5]
    assert views["centre3"]["epochs"] == [4]
    assert views["top3"]["epochs"] == [2, 4, 5]
    assert views["around3"]["epochs"] == [4, 5, 6]
    early = epoch_views(_trajectory([0.10, 0.0, 0.0]), rules=("around3", "top5"))
    assert early["around3"]["epochs"] == [1, 2]
    assert early["top5"]["epochs"] == [1, 2, 3]
    with pytest.raises(ValueError):
        epoch_views(history, rules=("median3",))


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
