from __future__ import annotations

import numpy as np

from brazil_rv.preprocessing.p1_features import (
    P1_FEATURES,
    build_security_library,
    edge_spread,
)


def _bars(date_count: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.zeros((date_count, 405, 5), dtype=np.float64)
    observed = np.ones((date_count, 405), dtype=bool)
    accepted = np.ones(date_count, dtype=bool)
    for day in range(date_count):
        close = 10.0 * np.exp(0.0001 * np.arange(405) + 0.001 * day)
        raw[day, :, 0] = close * 0.9999
        raw[day, :, 1] = close * 1.0003
        raw[day, :, 2] = close * 0.9997
        raw[day, :, 3] = close
        raw[day, :, 4] = 1000 + 3 * np.arange(405) + 10 * day
    return raw, observed, accepted


def test_edge_is_price_scale_invariant_and_requires_three_observations() -> None:
    open_ = np.array([10.0, 10.1, 10.0, 10.2, 10.1])
    high = open_ + 0.1
    low = open_ - 0.1
    close = open_ + np.array([0.02, -0.03, 0.04, -0.02, 0.01])
    expected = edge_spread(open_, high, low, close)
    assert np.isfinite(expected)
    np.testing.assert_allclose(
        edge_spread(open_ * 7, high * 7, low * 7, close * 7), expected, atol=1e-14
    )
    assert np.isnan(edge_spread(open_[:2], high[:2], low[:2], close[:2]))


def test_p1_library_is_causal_to_future_and_post_decision_mutations() -> None:
    raw, observed, accepted = _bars()
    baseline, baseline_mask = build_security_library(raw, observed, accepted)
    mutated = raw.copy()
    mutated[26, :, :4] *= 1.4
    mutated[25, 15:, :4] *= 1.2
    candidate, candidate_mask = build_security_library(mutated, observed, accepted)
    np.testing.assert_array_equal(candidate_mask[:25], baseline_mask[:25])
    np.testing.assert_allclose(candidate[:25], baseline[:25])
    np.testing.assert_array_equal(candidate_mask[25, 0], baseline_mask[25, 0])
    np.testing.assert_allclose(candidate[25, 0], baseline[25, 0])
    assert baseline.shape[-1] == len(P1_FEATURES)
    assert baseline_mask[25].any()


def test_missing_bars_remain_masked_without_stale_endpoints() -> None:
    raw, observed, accepted = _bars()
    observed[25, 14] = False
    values, mask = build_security_library(raw, observed, accepted)
    reversal = P1_FEATURES.index("vwap_reversal_15m_cs")
    assert not mask[25, 0, reversal]
    assert values[25, 0, reversal] == 0.0
