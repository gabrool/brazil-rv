"""Protect the exact-prefix selector against future-history selection."""

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
