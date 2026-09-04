from __future__ import annotations

import numpy as np

from brazil_rv.v2.baselines import build_baselines, rank_gaussianize


def _prices() -> tuple[np.ndarray, np.ndarray]:
    sessions = np.arange(270)[:, None]
    returns = np.asarray([1.001, 1.002, 1.003])[None]
    close = np.power(returns, sessions)
    return close, np.ones_like(close, dtype=bool)


def test_rank_gaussianize_is_centered_and_tie_aware() -> None:
    values = np.asarray([[1.0, 2.0, 2.0, 4.0]])
    mask = np.ones_like(values, dtype=bool)
    transformed = rank_gaussianize(values, mask)
    assert abs(float(transformed.mean())) < 1e-7
    assert transformed[0, 1] == transformed[0, 2]
    assert np.isfinite(transformed).all()


def test_baseline_signs_windows_and_output_contract() -> None:
    close, observed = _prices()
    active = np.ones_like(observed)
    unresolved = np.zeros_like(observed)
    panels = build_baselines(close, observed, active, unresolved)
    assert set(panels) == {
        "reversal_5",
        "reversal_21",
        "momentum_12_1",
        "reversal_5_momentum_12_1_blend",
    }
    for panel in panels.values():
        assert panel.scores.shape == (270, 3, 5)
        assert panel.score_mask.shape == panel.scores.shape
    assert panels["reversal_5"].scores[260, :, 0].tolist() == [2.0, 1.0, 0.0]
    assert panels["momentum_12_1"].scores[260, :, 0].tolist() == [0.0, 1.0, 2.0]
    assert not panels["momentum_12_1"].score_mask[252].any()
    assert panels["momentum_12_1"].score_mask[253].all()
    assert (
        build_baselines(close, observed, active, unresolved, slow_lag=0)[
            "momentum_12_1"
        ]
        .score_mask[252]
        .all()
    )


def test_baselines_are_causal_and_keep_missing_endpoint_masked() -> None:
    close, observed = _prices()
    active = np.ones_like(observed)
    unresolved = np.zeros_like(observed)
    reference = build_baselines(close, observed, active, unresolved)
    changed = close.copy()
    changed[260:] *= 100.0
    actual = build_baselines(changed, observed, active, unresolved)
    for name in reference:
        assert np.array_equal(reference[name].scores[:261], actual[name].scores[:261])
        assert np.array_equal(
            reference[name].score_mask[:261], actual[name].score_mask[:261]
        )
    observed[254, 0] = False
    missing = build_baselines(close, observed, active, unresolved)
    assert not missing["reversal_5"].score_mask[260, 0].any()


def test_baselines_mask_returns_crossing_ambiguous_actions() -> None:
    close, observed = _prices()
    active = np.ones_like(observed)
    unresolved = np.zeros_like(observed)
    unresolved[254, 0] = True

    panels = build_baselines(close, observed, active, unresolved)

    # Fine/evaluation baselines at t=260 end at t-1=259. The five-session
    # reversal starts on 254, so the event is not crossed; 21 and 252 are.
    assert panels["reversal_5"].score_mask[260, 0].all()
    assert not panels["reversal_21"].score_mask[260, 0].any()
    assert panels["momentum_12_1"].score_mask[260, 0].all()
    # An event inside (254, 259] invalidates the five-session return too.
    unresolved[255, 0] = True
    crossed = build_baselines(close, observed, active, unresolved)
    assert not crossed["reversal_5"].score_mask[260, 0].any()
    unresolved[100, 1] = True
    crossed = build_baselines(close, observed, active, unresolved)
    assert not crossed["momentum_12_1"].score_mask[260, 1].any()
