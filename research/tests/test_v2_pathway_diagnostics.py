import numpy as np

from brazil_rv.v2.pathway_diagnostics import (
    fit_ridge,
    leave_one_out,
    peer_features,
    transform,
)


def test_peer_means_exclude_self_and_unknown_observations():
    values = np.array([2.0, 6.0, 100.0, 200.0])
    result = leave_one_out(
        values, np.array([True, True, False, True]), np.array([True, True, True, False])
    )
    np.testing.assert_allclose(result, [6.0, 2.0, 4.0, 4.0])


def test_peer_features_are_decision_causal_and_undefined_without_source():
    days, names = 150, 16
    dates = np.arange(np.datetime64("2010-01-01"), np.datetime64("2010-01-01") + days)
    rng = np.random.default_rng(11)
    magnitude = rng.normal(size=(days, names, 4)).astype(np.float32)
    magnitude[..., 1] = 0.02
    known = np.ones_like(magnitude, bool)
    active = np.ones((days, names), bool)
    sector = np.full((days, names), "a", object)
    market = rng.normal(size=(days, names, 3)).astype(np.float32)
    market_valid = np.ones_like(market, bool)
    fields = ["adr_return_gap_1", "exposure_fx", "adr_listed_flag"]
    before, _ = peer_features(
        dates, active, magnitude, known, sector, market, market_valid, fields
    )
    magnitude[120:] *= 100
    market[120:] *= 100
    after, _ = peer_features(
        dates, active, magnitude, known, sector, market, market_valid, fields
    )
    np.testing.assert_array_equal(before[:120], after[:120])
    known[:, :, :2] = False
    missing, _ = peer_features(
        dates, active, magnitude, known, sector, market, market_valid, fields
    )
    assert np.isnan(missing[:, :, :15]).all()


def test_ridge_preprocessing_is_fixed_before_evaluation_and_missing_is_explicit():
    rng = np.random.default_rng(29)
    x = rng.normal(size=(200, 3))
    x[::3, 1] = np.nan
    y = np.nan_to_num(x).sum(1)
    model = fit_ridge(x, y, np.repeat(np.arange(20), 10), 0.01)
    saved = {k: v.copy() for k, v in model.items()}
    transformed = transform(np.array([[1e9, np.nan, 0.0]]), model)
    assert np.isfinite(transformed).all()
    assert transformed[0, 1] == 0 and transformed[0, 4] == 0
    for key in saved:
        np.testing.assert_array_equal(saved[key], model[key])
