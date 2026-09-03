from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from brazil_rv.v2 import splits


def _weekdays(start: date, end: date) -> tuple[date, ...]:
    values: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            values.append(current)
        current += timedelta(days=1)
    return tuple(values)


def test_development_folds_have_exact_windows_and_75_session_embargo() -> None:
    calendar = _weekdays(date(2021, 8, 16), date(2024, 12, 30))
    folds = splits.development_folds(calendar)

    assert [fold.name for fold in folds] == ["F1", "F2", "F3"]
    assert [(fold.selection_dates[0], fold.selection_dates[-1]) for fold in folds] == [
        (date(2023, 7, 3), date(2023, 12, 29)),
        (date(2024, 1, 2), date(2024, 6, 28)),
        (date(2024, 7, 1), date(2024, 12, 30)),
    ]
    positions = {value: index for index, value in enumerate(calendar)}
    for fold in folds:
        assert len(fold.embargo_dates) == 75
        assert (
            positions[fold.fit_dates[-1]] + max(splits.HORIZONS)
            < positions[fold.selection_dates[0]]
        )
        assert fold.fit_dates[-1] < fold.embargo_dates[0]
        assert fold.embargo_dates[-1] < fold.selection_dates[0]


def test_development_fold_uses_sessions_inside_a_closed_date_boundary() -> None:
    calendar = tuple(
        value
        for value in _weekdays(date(2021, 8, 16), date(2024, 12, 30))
        if value != date(2023, 12, 29)
    )

    folds = splits.development_folds(calendar)

    assert folds[0].selection_dates[0] == date(2023, 7, 3)
    assert folds[0].selection_dates[-1] == date(2023, 12, 28)
    assert date(2024, 1, 2) not in folds[0].selection_dates


def test_block_parity_is_window_local_and_complementary() -> None:
    dates = _weekdays(date(2024, 1, 2), date(2024, 1, 18))
    forward, reverse = splits.block_parity_directions(dates)

    assert forward.selection_dates == dates[:5] + dates[10:]
    assert forward.evaluation_dates == dates[5:10]
    assert reverse.selection_dates == forward.evaluation_dates
    assert reverse.evaluation_dates == forward.selection_dates
    assert set(forward.selection_dates) | set(forward.evaluation_dates) == set(dates)


def test_target_masks_are_clipped_to_the_requested_window() -> None:
    dates = _weekdays(date(2024, 1, 2), date(2024, 1, 17))
    window = dates[:8]
    target_mask = np.ones((len(dates), 2, len(splits.HORIZONS)), dtype=bool)

    clipped = splits.mask_targets_to_window(
        target_mask,
        calendar_dates=dates,
        window_dates=window,
    )

    for index, horizon in enumerate(splits.HORIZONS):
        if horizon < len(window):
            assert clipped[: 8 - horizon, :, index].all()
            assert not clipped[8 - horizon :, :, index].any()
        else:
            assert not clipped[:, :, index].any()
    assert target_mask.all()


def test_official_window_requires_and_records_registration(tmp_path) -> None:
    root = tmp_path / "research" / "preregistrations"
    root.mkdir(parents=True)
    registration = root / "official_v2.md"
    registration.write_text("frozen registration\n", encoding="utf-8")
    dates = (date(2025, 1, 2), date(2025, 1, 3))

    with pytest.raises(PermissionError, match="preregistration"):
        splits.authorize_dates(dates, purpose="evaluation")
    ledger = splits.authorize_dates(
        dates,
        purpose="evaluation",
        registration_path=registration,
        preregistration_root=root,
    )

    assert ledger.official_validation_accessed is True
    assert ledger.test_accessed is False
    assert ledger.registration is not None
    assert ledger.registration.path == registration.resolve()
    assert len(ledger.registration.sha256) == 64

    for purpose in ("training", "selection"):
        with pytest.raises(PermissionError, match="never be used"):
            splits.authorize_dates(
                dates,
                purpose=purpose,
                registration_path=registration,
                preregistration_root=root,
            )


def test_test_dates_are_refused_and_no_whole_array_loader_is_public() -> None:
    assert not hasattr(splits, "load_authorized_array")
    with pytest.raises(ValueError, match="unconditionally sealed"):
        splits.authorize_dates(
            (date(2026, 1, 2),),
            purpose="evaluation",
        )


def test_registration_cannot_escape_preregistration_root(tmp_path) -> None:
    root = tmp_path / "preregistrations"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("not registered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="inside research/preregistrations"):
        splits.authorize_dates(
            (date(2025, 1, 2),),
            purpose="evaluation",
            registration_path=outside,
            preregistration_root=root,
        )


def test_contiguous_session_axis_uses_canonical_positions() -> None:
    dates = _weekdays(date(2024, 1, 2), date(2024, 1, 8))

    normalized = splits.validate_contiguous_session_axis(
        dates, np.arange(20, 20 + len(dates), dtype=np.int32)
    )
    np.testing.assert_array_equal(normalized, np.arange(20, 25))

    with pytest.raises(ValueError, match="gappy or parity-only"):
        splits.validate_contiguous_session_axis(
            dates, np.asarray([20, 22, 24, 26, 28], dtype=np.int64)
        )
