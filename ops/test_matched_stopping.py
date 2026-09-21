"""Protect the exact-prefix selector against future-history selection."""

import numpy as np

from replay_data_refits import forecast_identity
from run_matched_stopping import stopped_selection


def test_future_epochs_cannot_change_early_stop():
    history = [
        dict(epoch=i + 1, selection=dict(mean_ic=x))
        for i, x in enumerate([0.1, 0.1, 0.10005, 0.09, 0.08, 0.07, 0.5])
    ]
    early = stopped_selection(history, 5, 0.0001)
    assert early == dict(
        selected_epoch=1, selection_ic=0.1, epochs_completed=6, stop_reason="patience"
    )
    assert stopped_selection(history[:6], 5, 0.0001) == early
    assert stopped_selection(history, 20, 0.0001)["selected_epoch"] == 7


def test_meaningful_improvement_resets_stale_and_earlier_ties_win():
    history = [
        dict(epoch=i + 1, selection=dict(mean_ic=x))
        for i, x in enumerate(
            [0.1, 0.09, 0.08, 0.1002, 0.1002, 0.1002, 0.1002, 0.1002, 0.1002, 0.9]
        )
    ]
    chosen = stopped_selection(history, 5, 0.0001)
    assert chosen["selected_epoch"] == 4 and chosen["epochs_completed"] == 9


def test_book_reuse_requires_same_forecasts_support_and_policy():
    panel = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    valid = np.ones((2, 3), dtype=bool)
    args = ("F2", 10000000, "11", "TE_all")
    original = forecast_identity(panel, valid, *args)
    assert forecast_identity(panel.copy(), valid.copy(), *args) == original
    changed = panel.copy()
    changed[0, 0, 0] = np.nextafter(np.float32(0), np.float32(1))
    assert forecast_identity(changed, valid, *args) != original
    missing = valid.copy()
    missing[0, 0] = False
    assert forecast_identity(panel, missing, *args) != original
    assert forecast_identity(panel, valid, *args[:3], "C6") != original
    assert forecast_identity(panel, valid, "F3", *args[1:]) != original
    assert forecast_identity(panel, valid, args[0], 1000000, *args[2:]) != original
