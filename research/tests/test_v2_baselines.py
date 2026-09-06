from __future__ import annotations

import numpy as np

from brazil_rv.v2.baselines import (
    build_baselines,
    build_store_baselines,
    rank_gaussianize,
)


def _shareholder_wealth() -> tuple[np.ndarray, np.ndarray]:
    sessions = np.arange(270)[:, None]
    returns = np.asarray([1.001, 1.002, 1.003])[None]
    close = np.power(returns, sessions)
    return close, np.ones_like(close, dtype=bool)


def _build(
    wealth_close: np.ndarray,
    wealth_valid: np.ndarray,
    active: np.ndarray,
    ambiguous: np.ndarray,
):
    volatility = np.broadcast_to(
        np.asarray([0.03, 0.02, 0.01]), wealth_close.shape
    ).copy()
    return build_baselines(
        wealth_close,
        wealth_valid,
        active,
        ambiguous,
        volatility,
    )


def test_rank_gaussianize_is_centered_and_tie_aware() -> None:
    values = np.asarray([[1.0, 2.0, 2.0, 4.0]])
    mask = np.ones_like(values, dtype=bool)
    transformed = rank_gaussianize(values, mask)
    assert abs(float(transformed.mean())) < 1e-7
    assert transformed[0, 1] == transformed[0, 2]
    assert np.isfinite(transformed).all()


def test_baseline_signs_windows_and_output_contract() -> None:
    wealth_close, wealth_valid = _shareholder_wealth()
    active = np.ones_like(wealth_valid)
    unresolved = np.zeros_like(wealth_valid)
    panels = _build(wealth_close, wealth_valid, active, unresolved)
    assert set(panels) == {
        "reversal_5",
        "reversal_21",
        "momentum_12_1",
        "reversal_5_momentum_12_1_blend",
        "inverse_volatility_20",
    }
    for panel in panels.values():
        assert panel.scores.shape == (270, 3, 5)
        assert panel.score_mask.shape == panel.scores.shape
    assert panels["reversal_5"].scores[260, :, 0].tolist() == [2.0, 1.0, 0.0]
    assert panels["momentum_12_1"].scores[260, :, 0].tolist() == [0.0, 1.0, 2.0]
    assert not panels["momentum_12_1"].score_mask[252].any()
    assert panels["momentum_12_1"].score_mask[253].all()


def test_inverse_volatility_uses_decision_row_canonical_sigma_without_relag() -> None:
    wealth_close, wealth_valid = _shareholder_wealth()
    active = np.ones_like(wealth_valid)
    action_boundary = np.zeros_like(wealth_valid)
    sigma = np.full_like(wealth_close, np.nan, dtype=np.float64)
    sigma[0] = [0.03, 0.01, 0.02]
    sigma[1] = [0.01, 0.03, 0.02]

    panel = build_baselines(wealth_close, wealth_valid, active, action_boundary, sigma)[
        "inverse_volatility_20"
    ]

    assert panel.score_mask[0].all()
    assert panel.scores[0, :, 0].tolist() == [0.0, 2.0, 1.0]
    assert panel.scores[1, :, 0].tolist() == [2.0, 0.0, 1.0]


def test_store_baseline_adapter_uses_shared_decision_axis() -> None:
    wealth_close, wealth_valid = _shareholder_wealth()
    active = np.ones_like(wealth_valid)
    active[260, 0] = False
    boundary = np.zeros_like(wealth_valid)
    sigma = np.broadcast_to(
        np.asarray([0.03, 0.02, 0.01]), wealth_close.shape
    ).copy()
    arrays = {
        "active": active,
        "shareholder_wealth_close": wealth_close,
        "shareholder_wealth_valid": wealth_valid,
        "decision_action_boundary_mask": boundary,
        "target_scale_sigma": sigma,
    }

    class Store:
        dates = np.arange(270).astype("datetime64[D]")
        isins = ("BR1", "BR2", "BR3")
        manifest = {"feature_names": {}}

        @staticmethod
        def read(name, indices):
            return arrays[name][indices]

    indices = np.arange(270, dtype=np.int64)
    actual = build_store_baselines(Store(), indices)
    expected = build_baselines(wealth_close, wealth_valid, active, boundary, sigma)

    for name in expected:
        np.testing.assert_array_equal(actual[name].scores, expected[name].scores)
        np.testing.assert_array_equal(
            actual[name].score_mask, expected[name].score_mask
        )
    assert not actual["reversal_5"].score_mask[260, 0].any()


def test_baselines_are_causal_and_keep_missing_endpoint_masked() -> None:
    wealth_close, wealth_valid = _shareholder_wealth()
    active = np.ones_like(wealth_valid)
    action_boundary = np.zeros_like(wealth_valid)
    reference = _build(wealth_close, wealth_valid, active, action_boundary)
    changed = wealth_close.copy()
    changed[260:] *= 100.0
    actual = _build(changed, wealth_valid, active, action_boundary)
    for name in reference:
        assert np.array_equal(reference[name].scores[:261], actual[name].scores[:261])
        assert np.array_equal(
            reference[name].score_mask[:261], actual[name].score_mask[:261]
        )
    wealth_valid[254, 0] = False
    missing = _build(wealth_close, wealth_valid, active, action_boundary)
    assert not missing["reversal_5"].score_mask[260, 0].any()


def test_baselines_mask_returns_crossing_decision_known_action_boundaries() -> None:
    wealth_close, wealth_valid = _shareholder_wealth()
    active = np.ones_like(wealth_valid)
    action_boundary = np.zeros_like(wealth_valid)
    action_boundary[254, 0] = True

    panels = _build(wealth_close, wealth_valid, active, action_boundary)

    # Fine/evaluation baselines at t=260 end at t-1=259. The five-session
    # reversal starts on 254, so the event is not crossed; 21 and 252 are.
    assert panels["reversal_5"].score_mask[260, 0].all()
    assert not panels["reversal_21"].score_mask[260, 0].any()
    assert panels["momentum_12_1"].score_mask[260, 0].all()
    # An event inside (254, 259] invalidates the five-session return too.
    action_boundary[255, 0] = True
    crossed = _build(wealth_close, wealth_valid, active, action_boundary)
    assert not crossed["reversal_5"].score_mask[260, 0].any()
    action_boundary[100, 1] = True
    crossed = _build(wealth_close, wealth_valid, active, action_boundary)
    assert not crossed["momentum_12_1"].score_mask[260, 1].any()
