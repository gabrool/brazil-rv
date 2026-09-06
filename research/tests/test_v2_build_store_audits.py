from datetime import date, datetime, timezone

import numpy as np
import polars as pl
import pytest

from brazil_rv.v2.build_store import (
    _common_state_diagnostic_panel,
    _route_decision_known_continuations,
    _target_validity_tables,
)
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
    yearly, survival = _target_validity_tables(
        dates, valid, active, observed, family="primary"
    )
    assert yearly.height == 2 * len(HORIZONS)
    assert set(survival.get_column("group")) == {
        "delisted_within_panel",
        "survives_to_final_year",
    }


def test_target_validity_audit_rejects_survivorship_gap_over_ten_points() -> None:
    dates, active, observed, valid = _validity_fixture()
    valid[:, 0] = False
    with pytest.raises(ValueError, match="survivor-skewed"):
        _target_validity_tables(dates, valid, active, observed, family="primary")


def test_common_state_panel_uses_only_completed_daily_rows_and_raw_sigma() -> None:
    names = 20
    wealth = np.vstack(
        (
            np.full(names, 100.0),
            np.linspace(99.0, 103.0, names),
            np.full(names, 999.0),
        )
    )
    sigma = np.vstack(
        (
            np.full(names, 9.0),
            np.full(names, 8.0),
            np.linspace(0.01, 0.03, names),
        )
    )
    expected_returns = np.log(wealth[1] / wealth[0])
    values, valid, table = _common_state_diagnostic_panel(
        wealth,
        np.ones_like(wealth, dtype=np.bool_),
        np.zeros_like(wealth, dtype=np.bool_),
        np.ones_like(wealth, dtype=np.bool_),
        sigma,
        np.asarray([2]),
        np.asarray(["2024-01-02", "2024-01-03", "2024-01-04"], dtype="datetime64[D]"),
        minimum_names=20,
    )

    assert valid.all()
    np.testing.assert_allclose(values[0, 0], np.median(expected_returns))
    np.testing.assert_allclose(values[0, 1], 0.02)
    np.testing.assert_allclose(values[0, 2], np.std(expected_returns, ddof=0))
    assert table[0, "return_support_names"] == 20
    assert table[0, "volatility_support_names"] == 20


def _continuation_link(*, first_known_at: datetime) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "predecessor_isin": ["BRTESTACNOR1"],
            "successor_isin": ["BRTESTACNPR0"],
            "successor_first_date": [date(2024, 1, 4)],
            "first_known_at": [first_known_at],
            "shares_received_per_prior_share": [2.0],
            "cash_entitlement_per_prior_share": [5.0],
        }
    )


def test_decision_known_conversion_routes_each_feature_family_by_its_units() -> None:
    dates = np.arange(np.datetime64("2024-01-02"), np.datetime64("2024-01-07"))
    decisions = tuple(
        datetime.combine(
            value.astype(object), datetime.min.time(), timezone.utc
        ).replace(hour=18, minute=45)
        for value in dates
    )
    observed = np.asarray(
        [[True, False], [True, False], [False, True], [False, True], [False, True]]
    )
    raw_close = np.asarray(
        [
            [100.0, np.nan],
            [110.0, np.nan],
            [np.nan, 55.0],
            [np.nan, 56.0],
            [np.nan, 57.0],
        ]
    )
    volume = np.asarray(
        [[10.0, np.nan], [20.0, np.nan], [np.nan, 30.0], [np.nan, 40.0], [np.nan, 50.0]]
    )
    trades = volume / 10.0
    wealth = np.asarray(
        [[100.0, 0.0], [110.0, 0.0], [115.0, 55.0], [117.0, 56.0], [119.0, 57.0]]
    )
    wealth_valid = observed.copy()
    routed = _route_decision_known_continuations(
        dates=dates,
        isins=("BRTESTACNOR1", "BRTESTACNPR0"),
        links=_continuation_link(
            first_known_at=datetime(2024, 1, 3, 12, tzinfo=timezone.utc)
        ),
        decision_timestamps=decisions,
        raw_close=raw_close,
        volume_brl=volume,
        trades=trades,
        observed=observed,
        trade_observed=observed,
        activity_valid=observed,
        ambiguous_action=np.zeros_like(observed),
        shareholder_wealth_arrays=(wealth, wealth_valid),
    )

    np.testing.assert_allclose(wealth[:, 1], [100.0, 110.0, 115.0, 117.0, 119.0])
    np.testing.assert_allclose(routed.close_brl[:2, 1], [50.0, 55.0])
    np.testing.assert_allclose(routed.volume_brl[:2, 1], [10.0, 20.0])
    np.testing.assert_allclose(routed.trades[:2, 1], [1.0, 2.0])
    assert routed.observed[:, 1].all()
    assert routed.trade_observed[:, 1].all()
    assert routed.activity_valid[:, 1].all()
    assert routed.claim_owner.sum(axis=1).tolist() == [1, 1, 1, 1, 1]
    assert routed.claim_owner[:, 0].tolist() == [True, True, False, False, False]


def test_late_verified_conversion_does_not_rewrite_earlier_feature_history() -> None:
    dates = np.arange(np.datetime64("2024-01-02"), np.datetime64("2024-01-07"))
    decisions = tuple(
        datetime.combine(
            value.astype(object), datetime.min.time(), timezone.utc
        ).replace(hour=18, minute=45)
        for value in dates
    )
    observed = np.asarray(
        [[True, False], [True, False], [False, True], [False, True], [False, True]]
    )
    close = np.asarray(
        [
            [100.0, np.nan],
            [110.0, np.nan],
            [np.nan, 55.0],
            [np.nan, 56.0],
            [np.nan, 57.0],
        ]
    )
    wealth = np.where(observed, np.nan_to_num(close), 0.0)
    before = wealth.copy()
    routed = _route_decision_known_continuations(
        dates=dates,
        isins=("BRTESTACNOR1", "BRTESTACNPR0"),
        links=_continuation_link(
            first_known_at=datetime(2024, 1, 6, 12, tzinfo=timezone.utc)
        ),
        decision_timestamps=decisions,
        raw_close=close,
        volume_brl=np.where(observed, 10.0, np.nan),
        trades=np.where(observed, 1.0, np.nan),
        observed=observed,
        trade_observed=observed,
        activity_valid=observed,
        ambiguous_action=np.zeros_like(observed),
        shareholder_wealth_arrays=(wealth,),
    )

    np.testing.assert_array_equal(wealth, before)
    np.testing.assert_array_equal(routed.observed, observed)
    assert routed.claim_owner[:4, 0].all()
    assert not routed.claim_owner[4, 0]
