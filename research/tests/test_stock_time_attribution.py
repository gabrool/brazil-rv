from __future__ import annotations

import numpy as np

from brazil_rv.modeling.analyze_stock_time_attribution import (
    AttributionInputs,
    horizon_attribution,
    opening_attribution,
    stock_attribution,
    time_of_day_attribution,
)


def _inputs() -> AttributionInputs:
    rng = np.random.default_rng(20260813)
    predictions = rng.normal(size=(6, 30, 3)).astype(np.float32)
    targets = (predictions + rng.normal(scale=0.2, size=predictions.shape)).astype(
        np.float32
    )
    returns = rng.normal(scale=0.01, size=predictions.shape).astype(np.float32)
    mask = np.ones_like(predictions, dtype=bool)
    mask[0, 0, 0] = False
    return AttributionInputs(
        "run-a",
        predictions,
        targets,
        returns,
        mask,
        np.array([1, 1, 1, 2, 2, 2]),
        np.array([0, 1, 2, 0, 1, 2]),
        tuple(f"SEC-{slot}" for slot in range(30)),
        np.array([-2.0, -2.0, -2.0, 2.0, 2.0, 2.0]),
        (-1.0, 1.0),
    )


def test_stock_attribution_reports_contribution_skill_coverage_and_returns() -> None:
    frame = stock_attribution(_inputs())
    assert frame.height == 30 * 3
    assert set(frame.columns) >= {
        "security_id",
        "horizon_minutes",
        "mean_spearman_contribution",
        "time_series_rank_skill",
        "coverage",
        "mean_raw_return",
    }
    assert frame.get_column("security_id").n_unique() == 30


def test_time_horizon_and_opening_reports_keep_economic_diagnostics() -> None:
    inputs = _inputs()
    time = time_of_day_attribution(inputs)
    horizon = horizon_attribution(inputs)
    opening = opening_attribution(inputs)
    assert time.height == 3 * 3
    assert horizon.height == 3
    assert opening.get_column("opening_regime").sort().to_list() == [
        "negative_tail",
        "negative_tail",
        "negative_tail",
        "positive_tail",
        "positive_tail",
        "positive_tail",
    ]
    for frame in (time, horizon, opening):
        assert set(frame.columns) >= {
            "mean_spearman_ic",
            "mean_top_minus_bottom",
            "mean_one_way_turnover",
            "label_coverage",
        }


def test_analysis_surface_is_validation_only() -> None:
    from brazil_rv.modeling.analyze_stock_time_attribution import parse_args

    args = parse_args(["--run-dir", "run", "--output-dir", "output"])
    assert not hasattr(args, "split")
