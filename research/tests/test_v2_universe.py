import numpy as np
import pytest

from brazil_rv.v2.universe import build_daily_universe, v1_pit_coverage_table


def test_universe_uses_strictly_prior_sessions() -> None:
    close = np.full((8, 2), 10.0)
    volume = np.full((8, 2), 100.0)
    observed = np.ones((8, 2), dtype=bool)
    result = build_daily_universe(
        close,
        volume,
        observed,
        prior_sessions=3,
        minimum_traded=2,
        minimum_median_volume_brl=50.0,
        minimum_prior_close_brl=1.0,
        minimum_history_sessions=3,
    )
    assert result.active[3:].all()
    changed_close = close.copy()
    changed_volume = volume.copy()
    changed_seen = observed.copy()
    changed_close[5] = 0.0
    changed_volume[5] = 0.0
    changed_seen[5] = False
    mutated = build_daily_universe(
        changed_close,
        changed_volume,
        changed_seen,
        prior_sessions=3,
        minimum_traded=2,
        minimum_median_volume_brl=50.0,
        minimum_prior_close_brl=1.0,
        minimum_history_sessions=3,
    )
    np.testing.assert_array_equal(result.active[5], mutated.active[5])


def test_v1_subset_audit_allows_dynamic_membership_but_requires_each_name() -> None:
    dates = np.asarray(["2021-08-16", "2021-08-17", "2021-08-18"], dtype="datetime64[D]")
    isins = ("BRTESTACNOR1", "BRTESTACNPR0", "BRANOTHRNOR1")
    active = np.asarray(
        [[True, False, True], [False, True, True], [True, False, False]]
    )
    table = v1_pit_coverage_table(
        dates, isins, active, ("BRTESTACNOR1", "BRTESTACNPR0")
    )
    assert table.get_column("active_v1_count").to_list() == [1, 1, 1]
    assert table.get_column("mapped_v1_count").to_list() == [2, 2, 2]

    active[:, 1] = False
    with pytest.raises(ValueError, match="never PIT-active"):
        v1_pit_coverage_table(
            dates, isins, active, ("BRTESTACNOR1", "BRTESTACNPR0")
        )
