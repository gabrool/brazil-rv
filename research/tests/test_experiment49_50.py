from __future__ import annotations

import numpy as np
import polars as pl

from brazil_rv.modeling.experiment49_50 import (
    _combined_book,
    _rank_linear_weights,
    _spread_matrix,
    deploy_next_generation,
    keep_head15,
)
from brazil_rv.preprocessing.contract import DECISION_EQUITY_INDICES
from brazil_rv.preprocessing.economics_targets import (
    causal_trailing_adv,
    exact_alternative_returns,
    roll_covariance_inputs,
    roll_effective_spread,
)


def test_alternative_labels_use_exact_frozen_endpoints() -> None:
    raw = np.zeros((1, 420, 5), dtype=np.float64)
    observed = np.zeros((1, 420), dtype=bool)
    entry = DECISION_EQUITY_INDICES[0]
    raw[0, entry, 0] = 100.0
    raw[0, entry - 1, 3] = 99.0
    raw[0, entry, 3] = 101.0
    raw[0, entry + 15, 0] = 102.0
    raw[0, entry + 14, 3] = 102.0
    raw[0, entry + 15, 3] = 104.0
    observed[0, [entry - 1, entry, entry + 14, entry + 15]] = True

    open_return, open_mask, mid_return, mid_mask = exact_alternative_returns(
        raw, observed
    )

    assert open_mask[0, 0, 0]
    assert np.isclose(open_return[0, 0, 0], np.log(102.0 / 100.0))
    assert mid_mask[0, 0, 0]
    assert np.isclose(mid_return[0, 0, 0], np.log(103.0 / 100.0))

    mutated = raw.copy()
    mutated[0, entry + 16, 0] = 100_000.0
    changed, _, changed_mid, _ = exact_alternative_returns(mutated, observed)
    assert np.array_equal(open_return, changed)
    assert np.array_equal(mid_return, changed_mid)


def test_roll_spread_uses_negative_sample_serial_covariance() -> None:
    returns = np.asarray([0.01, -0.01, 0.01, -0.01], dtype=np.float64)
    closes = 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(returns))))
    raw = np.zeros((1, closes.size, 5), dtype=np.float64)
    raw[0, :, 3] = closes
    observed = np.ones((1, closes.size), dtype=bool)

    values = roll_covariance_inputs(raw, observed)
    covariance, spread = roll_effective_spread(*values)

    assert values[0] == 3
    assert covariance < 0.0
    assert np.isclose(spread, 2.0 * np.sqrt(-covariance))
    assert np.isnan(roll_effective_spread(2, 0.0, 0.0, 1.0)[1])


def test_trailing_adv_excludes_current_date_and_uses_observed_history() -> None:
    volume = np.asarray([[10.0], [0.0], [30.0], [50.0]])
    result = causal_trailing_adv(volume, window=2)

    assert np.isnan(result[0, 0])
    assert result[1, 0] == 10.0
    assert result[2, 0] == 10.0
    assert result[3, 0] == 20.0


def test_rank_linear_weights_are_dollar_neutral_and_two_times_gross() -> None:
    prediction = np.arange(30, dtype=np.float64)
    weights = _rank_linear_weights(prediction, np.ones(30, dtype=bool))

    assert np.isclose(weights.sum(), 0.0)
    assert np.isclose(np.abs(weights).sum(), 2.0)


def test_spread_matrix_leaves_post_schedule_dates_unmaterialized() -> None:
    schedule = pl.DataFrame(
        {
            "security_id": ["a", "b"],
            "quarter": ["2025Q2", "2025Q2"],
            "schedule_full_spread_fraction": [0.001, 0.002],
        }
    )
    dates = pl.DataFrame(
        {
            "date_idx": [0, 1],
            "trade_date": ["2025-06-30", "2025-07-01"],
        }
    )
    equities = pl.DataFrame({"equity_slot": [0, 1], "security_id": ["a", "b"]})

    values = _spread_matrix(schedule, dates, equities)

    assert np.array_equal(values[0], np.asarray([0.001, 0.002]))
    assert np.isnan(values[1]).all()


def test_combined_book_uses_only_prior_days_for_risk_weights() -> None:
    dates = np.arange(25)
    left = {
        "dates": dates,
        "gross": np.linspace(-0.02, 0.02, 25),
        "net": np.linspace(-0.01, 0.01, 25),
    }
    right = {
        "dates": dates,
        "gross": np.linspace(-0.002, 0.002, 25),
        "net": np.linspace(-0.001, 0.001, 25),
    }
    baseline = _combined_book(left, right)
    mutated = {**left, "gross": left["gross"].copy()}
    mutated["gross"][20] = 50_000.0
    changed = _combined_book(mutated, right)

    assert np.all(baseline["weights"][:20] == 0.5)
    assert np.array_equal(baseline["weights"][20], changed["weights"][20])


def test_frozen_verdict_and_deployment_boundaries_are_inclusive() -> None:
    assert keep_head15(
        mid_proxy_retention=(0.60, 0.60, 0.59),
        net_daily_return=(0.01, 0.02, 0.03),
        combined_sharpe=(1.0, 2.0, 0.0),
        comparator_sharpe=(1.0, 1.0, 1.0),
    )
    assert not keep_head15(
        mid_proxy_retention=(0.60, 0.60, 0.59),
        net_daily_return=(0.01, 0.0, 0.03),
        combined_sharpe=(1.0, 2.0, 0.0),
        comparator_sharpe=(1.0, 1.0, 1.0),
    )
    assert deploy_next_generation(-0.0005)
    assert not deploy_next_generation(-0.0005000001)
