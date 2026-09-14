import numpy as np
import pytest

from brazil_rv.v2.portfolio_program import frozen_fit_roster
from brazil_rv.v2.score import (
    validate_parent_forecast_dates,
    verify_reused_inference_source,
)


def test_frozen_roster_reuses_all_matching_fits_and_covers_all_seeds():
    jobs = frozen_fit_roster()
    assert len(jobs) == len(set(jobs)) == 102
    expected = {
        (arm, f"F{fold}", seed)
        for arm in ["S0", "TE_all", "C6"]
        for fold in range(1, 15)
        for seed in [11, 29, 47]
    }
    reused = {
        (arm, fold, seed)
        for arm in ["S0", "TE_all"]
        for fold in ["F2", "F6", "F10", "F14"]
        for seed in [11, 29, 47]
    }
    assert set(jobs) == expected - reused
    assert {fold for _, fold, _ in jobs[:9]} == {"F1"}


def test_parent_forecasts_require_chronological_out_of_fit_information():
    dates = np.arange("2016-01-01", "2018-01-01", dtype="datetime64[D]")
    boundary = np.searchsorted(dates, np.datetime64("2016-06-30"))
    validate_parent_forecast_dates(dates, np.arange(boundary + 11, len(dates)))
    with pytest.raises(ValueError, match="information boundary"):
        validate_parent_forecast_dates(dates, np.arange(boundary, len(dates)))
    with pytest.raises(ValueError, match="chronological"):
        validate_parent_forecast_dates(dates, np.arange(boundary + 11, len(dates), 2))


def test_reused_checkpoint_rejects_changed_inference_dependencies(monkeypatch):
    monkeypatch.setattr(
        "brazil_rv.v2.score._repository_commit_if_available", lambda: "a" * 40
    )
    monkeypatch.setattr(
        "brazil_rv.v2.score.subprocess.check_output",
        lambda *args, **kwargs: b"changed model",
    )
    with pytest.raises(ValueError, match="inference dependency changed"):
        verify_reused_inference_source("b" * 40)
