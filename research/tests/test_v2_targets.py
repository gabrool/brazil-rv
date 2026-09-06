import numpy as np

from brazil_rv.v2.corporate_actions import AlignedActionTerms
from brazil_rv.v2.targets import (
    build_economic_multi_day_targets,
    build_economic_multi_day_targets_into,
    build_to_close_target,
)


def _actions(
    q: np.ndarray,
    d: np.ndarray | None = None,
    resolved: np.ndarray | None = None,
) -> AlignedActionTerms:
    cash = np.zeros_like(q, dtype=np.float64) if d is None else np.asarray(d)
    known = np.ones_like(q, dtype=np.bool_) if resolved is None else resolved
    return AlignedActionTerms(
        shares_per_prior_share=np.asarray(q, dtype=np.float64),
        cash_per_prior_share=np.asarray(cash, dtype=np.float64),
        session_resolved=np.asarray(known, dtype=np.bool_),
        has_action=(np.asarray(q) != 1.0) | (np.asarray(cash) != 0.0),
    )


def test_economic_targets_distinguish_price_and_shareholder_returns() -> None:
    close = np.asarray(
        [
            [100.0, 100.0, 100.0],
            [55.0, 80.0, 50.0],
        ]
    )
    q = np.asarray([[1.0, 1.0, 1.0], [2.0, 1.0, 2.0]])
    d = np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    result = build_economic_multi_day_targets(
        close,
        np.ones_like(close, dtype=np.bool_),
        np.ones_like(close, dtype=np.bool_),
        np.full_like(close, 0.02),
        _actions(q, d),
        horizons=(1,),
    )
    np.testing.assert_allclose(
        result.price_simple_return[0, :, 0], [-0.45, -0.20, -0.50], atol=1e-7
    )
    np.testing.assert_allclose(
        result.shareholder_simple_return[0, :, 0], [0.10, -0.19, 0.01], atol=1e-7
    )
    np.testing.assert_allclose(
        result.terminal_wealth[0, :, 0], [1.10, 0.81, 1.01], atol=1e-7
    )
    assert result.entry_mark_type == "daily_last_trade_close_proxy"
    assert result.exit_mark_type == "daily_last_trade_close_proxy"


def test_holding_window_excludes_entry_day_event_and_does_not_rebook_payment() -> None:
    close = np.asarray([[50.0], [50.0], [50.0]])
    q = np.asarray([[2.0], [1.0], [1.0]])
    d = np.asarray([[1.0], [0.0], [0.0]])
    result = build_economic_multi_day_targets(
        close,
        np.ones_like(close, dtype=np.bool_),
        np.ones_like(close, dtype=np.bool_),
        np.full_like(close, 0.02),
        _actions(q, d),
        horizons=(1, 2),
    )
    # A close-t buyer is post-event and receives neither the event nor its later payment.
    assert result.shareholder_simple_return[0, 0, 0] == 0.0
    assert result.shareholder_simple_return[0, 0, 1] == 0.0


def test_exact_endpoints_survive_missing_intermediate_but_not_unknown_action_chain() -> None:
    close = np.asarray([[100.0], [np.nan], [110.0]])
    observed = np.asarray([[True], [False], [True]])
    result = build_economic_multi_day_targets(
        close,
        observed,
        np.ones_like(close, dtype=np.bool_),
        np.full_like(close, 0.02),
        _actions(np.ones_like(close)),
        horizons=(2,),
    )
    assert result.shareholder_valid[0, 0, 0]
    np.testing.assert_allclose(result.shareholder_simple_return[0, 0, 0], 0.10)

    unresolved = np.ones_like(close, dtype=np.bool_)
    unresolved[1, 0] = False
    unknown = build_economic_multi_day_targets(
        close,
        observed,
        np.ones_like(close, dtype=np.bool_),
        np.full_like(close, 0.02),
        _actions(np.ones_like(close), resolved=unresolved),
        horizons=(2,),
    )
    assert not unknown.shareholder_valid[0, 0, 0]


def test_verified_terminal_loss_is_valid_raw_and_bottom_ranked() -> None:
    close = np.asarray([[100.0, 100.0, 100.0], [np.nan, 100.0, 110.0]])
    observed = np.asarray([[True, True, True], [False, True, True]])
    q = np.asarray([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    result = build_economic_multi_day_targets(
        close,
        observed,
        np.ones_like(close, dtype=np.bool_),
        np.full_like(close, 0.02),
        _actions(q),
        horizons=(1,),
    )
    assert result.shareholder_valid[0, 0, 0]
    assert result.shareholder_simple_return[0, 0, 0] == -1.0
    assert result.terminal_wealth[0, 0, 0] == 0.0
    assert result.terminal_loss[0, 0, 0]
    assert result.shareholder_midrank[0, 0, 0] == 0.0
    assert result.normalized_residual[0, 0, 0] == -5.0
    assert result.primary_valid[0, 0, 0]


def test_nonfinite_zero_wealth_median_only_disables_normalized_cross_section() -> None:
    close = np.asarray([[100.0, 100.0], [np.nan, 100.0]])
    observed = np.asarray([[True, True], [False, True]])
    q = np.asarray([[1.0, 1.0], [0.0, 1.0]])
    result = build_economic_multi_day_targets(
        close,
        observed,
        np.ones_like(close, dtype=np.bool_),
        np.full_like(close, 0.02),
        _actions(q),
        horizons=(1,),
    )
    assert result.shareholder_valid[0, :, 0].all()
    assert not result.normalized_cross_section_valid[0, 0]
    assert not result.primary_valid[0, :, 0].any()
    assert np.isfinite(result.shareholder_simple_return[0, :, 0]).all()


def test_economic_target_follows_verified_successor_identity() -> None:
    close = np.asarray([[100.0, np.nan], [np.nan, 105.0], [np.nan, 110.0]])
    q = np.ones_like(close)
    successor = np.asarray([[0, 1], [1, 1], [0, 1]], dtype=np.int64)
    actions = _actions(q)
    actions = AlignedActionTerms(
        actions.shares_per_prior_share,
        actions.cash_per_prior_share,
        actions.session_resolved,
        actions.has_action,
        successor,
    )
    result = build_economic_multi_day_targets(
        close,
        np.isfinite(close),
        np.asarray([[True, False], [False, True], [False, True]]),
        np.full_like(close, 0.02),
        actions,
        horizons=(2,),
    )
    assert result.shareholder_valid[0, 0, 0]
    np.testing.assert_allclose(result.shareholder_simple_return[0, 0, 0], 0.10)


def test_raw_target_validity_is_independent_of_missing_sigma() -> None:
    close = np.array([[10.0, 10.0], [11.0, 12.0]])
    result = build_economic_multi_day_targets(
        close,
        np.ones_like(close, dtype=np.bool_),
        np.ones_like(close, dtype=np.bool_),
        np.full_like(close, np.nan),
        _actions(np.ones_like(close)),
        horizons=(1,),
    )
    assert result.shareholder_valid[0, :, 0].all()
    assert result.price_valid[0, :, 0].all()
    assert not result.primary_valid.any()
    assert result.shareholder_simple_return.dtype == np.float32


def test_economic_target_builder_streams_selected_rows_into_float32_destinations() -> None:
    close = np.arange(20, dtype=np.float64).reshape(5, 4) + 100.0
    shape = (2, 4, 2)
    floats = [np.empty(shape, dtype=np.float32) for _ in range(7)]
    masks = [np.empty(shape, dtype=np.bool_) for _ in range(4)]
    normalized_cross_section_valid = np.empty((2, 2), dtype=np.bool_)
    build_economic_multi_day_targets_into(
        raw_close=close,
        close_observed=np.ones_like(close, dtype=np.bool_),
        active=np.ones_like(close, dtype=np.bool_),
        sigma_asof=np.full_like(close, 0.02),
        actions=_actions(np.ones_like(close)),
        primary=floats[0],
        primary_valid=masks[0],
        normalized_residual=floats[1],
        normalized_cross_section_valid=normalized_cross_section_valid,
        shareholder_midrank=floats[2],
        shareholder_valid=masks[1],
        shareholder_simple_return=floats[3],
        terminal_wealth=floats[4],
        terminal_loss=masks[2],
        price_midrank=floats[5],
        price_valid=masks[3],
        price_simple_return=floats[6],
        source_rows=np.asarray([1, 3]),
        horizons=(1, 2),
    )
    assert masks[0][0].all()
    assert not masks[0][1, :, 1].any()
    np.testing.assert_allclose(floats[3][0, :, 0], close[2] / close[1] - 1.0)


def test_to_close_target_is_cross_sectionally_ranked() -> None:
    entry = np.array([[100.0, 100.0, 100.0]])
    close = np.array([[99.0, 100.0, 102.0]])
    result = build_to_close_target(
        entry,
        close,
        np.full_like(entry, 0.02),
        np.ones_like(entry, dtype=bool),
        np.ones_like(entry, dtype=bool),
    )
    np.testing.assert_array_equal(result.target[0], [0.0, 0.5, 1.0])


def test_residual_is_median_removed_before_name_specific_scaling() -> None:
    close = np.vstack((np.full(5, 100.0), np.full(5, 100.0 * np.exp(0.02))))
    sigma = np.tile(np.asarray([0.01, 0.02, 0.03, 0.04, 0.05]), (2, 1))
    result = build_economic_multi_day_targets(
        close,
        np.ones_like(close, dtype=np.bool_),
        np.ones_like(close, dtype=np.bool_),
        sigma,
        _actions(np.ones_like(close)),
        horizons=(1,),
    )
    np.testing.assert_array_equal(result.primary[0, :, 0], np.full(5, 0.5))


def test_target_uses_already_lagged_same_row_sigma_and_missing_path_stays_invalid() -> None:
    close = np.asarray([[100.0, 100.0, 100.0], [98.0, 100.0, 102.0]])
    sigma = np.full_like(close, 0.02)
    first = build_economic_multi_day_targets(
        close,
        np.ones_like(close, dtype=np.bool_),
        np.ones_like(close, dtype=np.bool_),
        sigma,
        _actions(np.ones_like(close)),
        horizons=(1,),
    )
    changed = sigma.copy()
    changed[0] = [0.5, 0.6, 0.7]
    second = build_economic_multi_day_targets(
        close,
        np.ones_like(close, dtype=np.bool_),
        np.ones_like(close, dtype=np.bool_),
        changed,
        _actions(np.ones_like(close)),
        horizons=(1,),
    )
    assert not np.array_equal(
        first.normalized_residual[0], second.normalized_residual[0]
    )
    observed = np.ones_like(close, dtype=np.bool_)
    observed[1, 0] = False
    missing = build_economic_multi_day_targets(
        close,
        observed,
        np.ones_like(close, dtype=np.bool_),
        sigma,
        _actions(np.ones_like(close)),
        horizons=(1,),
    )
    assert not missing.shareholder_valid[0, 0, 0]
