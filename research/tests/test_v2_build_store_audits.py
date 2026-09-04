import numpy as np
import pytest

from brazil_rv.v2.build_store import _target_validity_tables
from brazil_rv.v2.contract import HORIZONS


def _validity_fixture() -> tuple[np.ndarray, ...]:
    dates = np.concatenate(
        (
            np.arange(np.datetime64("2023-06-23"), np.datetime64("2023-12-20")),
            np.arange(np.datetime64("2024-01-02"), np.datetime64("2024-01-22")),
        )
    )
    active = np.ones((200, 2), dtype=bool)
    observed = np.ones_like(active)
    observed[180:, 0] = False
    active[180:, 0] = False
    valid = np.zeros((*active.shape, len(HORIZONS)), dtype=bool)
    for index, horizon in enumerate(HORIZONS):
        valid[: 180 - horizon, 0, index] = True
        valid[: 200 - horizon, 1, index] = True
    return dates, active, observed, valid


def test_target_validity_audit_reports_year_and_balanced_survival_groups() -> None:
    dates, active, observed, valid = _validity_fixture()
    yearly, survival = _target_validity_tables(dates, valid, active, observed)
    assert yearly.height == 2 * len(HORIZONS)
    assert set(survival.get_column("group")) == {
        "delisted_within_panel",
        "survives_to_final_year",
    }


def test_target_validity_audit_rejects_survivorship_gap_over_ten_points() -> None:
    dates, active, observed, valid = _validity_fixture()
    valid[:, 0] = False
    with pytest.raises(ValueError, match="survivor-skewed"):
        _target_validity_tables(dates, valid, active, observed)
