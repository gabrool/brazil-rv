from __future__ import annotations

import numpy as np

from brazil_rv.execution.experiment58 import (
    SCHEMA,
    TARGET_HORIZONS,
    _cell_weights,
    _daily_targets,
    _midrank_row,
    _part0_identity_matches,
    _sha256,
    _strict_through,
    _tail_weights,
    _turnover_with_terminal_liquidation,
    repair_artifact_paths,
)
from brazil_rv.execution.experiment56 import _atomic_json, _read_json


def test_midrank_is_tie_aware_and_excludes_invalid_names() -> None:
    values = np.asarray([3.0, 1.0, 1.0, 9.0])
    ranked = _midrank_row(values, np.asarray([True, True, True, False]))

    np.testing.assert_allclose(ranked[:3], [2 / 3, -1 / 3, -1 / 3])
    assert np.isnan(ranked[3])


def test_daily_targets_use_only_exact_requested_horizon_endpoint() -> None:
    days = 24
    names = 20
    close = np.arange(1, days + 1, dtype=np.float64)[:, None] * np.linspace(
        1.0, 2.0, names
    )
    close += np.linspace(-0.05, 0.05, names)[None, :] * np.arange(days)[:, None] ** 2
    sigma = np.full_like(close, 0.02)
    active = np.ones_like(close, dtype=bool)

    before, _ = _daily_targets(close, sigma, active)
    mutated = close.copy()
    mutated[12:] *= np.linspace(0.7, 1.3, names)
    after, _ = _daily_targets(mutated, sigma, active)

    for index, horizon in enumerate(TARGET_HORIZONS):
        np.testing.assert_array_equal(
            before[index, : 12 - horizon], after[index, : 12 - horizon]
        )


def test_tail_weights_are_equal_weight_dollar_neutral() -> None:
    scores = np.arange(10, dtype=np.float64)
    weights = _tail_weights(scores, np.ones(10, dtype=bool), 3)

    assert np.isclose(weights.sum(), 0.0)
    assert np.isclose(np.abs(weights).sum(), 2.0)
    np.testing.assert_allclose(weights[:3], -1 / 3)
    np.testing.assert_allclose(weights[-3:], 1 / 3)


def test_rank_band_retains_prior_rank_until_strictly_crossed() -> None:
    signal = np.asarray(
        [
            [-1.0, -0.9, -0.2, 0.2, 0.9, 1.0],
            [-0.8, -1.0, -0.1, 0.1, 1.0, 0.8],
            [-0.4, -0.9, -0.1, 0.1, 0.9, 0.4],
        ]
    )
    close = np.ones_like(signal)
    active = np.ones_like(signal, dtype=bool)

    banded = _cell_weights(signal, close, active, 1, 0.3)
    full = _cell_weights(signal, close, active, 1, 0.0)

    np.testing.assert_array_equal(np.flatnonzero(banded[1]), [0, 5])
    np.testing.assert_array_equal(np.flatnonzero(full[1]), [1, 4])


def test_terminal_liquidation_is_charged_to_last_interval() -> None:
    weights = np.asarray([[1.0, -1.0], [1.0, -1.0]])
    turnover = _turnover_with_terminal_liquidation(weights)

    np.testing.assert_allclose(turnover.sum(axis=1), [2.0, 2.0])


def test_patient_limits_require_strict_through_price() -> None:
    direction = np.asarray([1, 1, -1, -1])
    low = np.asarray([99.0, 100.0, 0.0, 0.0])
    high = np.asarray([0.0, 0.0, 101.0, 100.0])
    limit = np.full(4, 100.0)

    np.testing.assert_array_equal(
        _strict_through(direction, low, high, limit),
        [True, False, True, False],
    )


def test_part0_identity_tolerance_is_only_float_representation_scale() -> None:
    retained = -53.98252152374261

    assert _part0_identity_matches(-53.9825207685452, retained)
    assert not _part0_identity_matches(retained + 1.1e-6, retained)


def test_artifact_path_repair_changes_only_verified_metadata(tmp_path) -> None:
    market = tmp_path / "daily_market"
    market.mkdir()
    values = market / "daily_close.npy"
    with values.open("wb") as output:
        np.save(output, np.asarray([[1.0]]), allow_pickle=False)
    stale = tmp_path / ".daily_market.tmp" / values.name
    manifest = {
        "schema": "EXPERIMENT58_DAILY_MARKET_V1",
        "artifacts": {
            values.name: {
                "path": str(stale),
                "bytes": values.stat().st_size,
                "sha256": _sha256(values),
            }
        },
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(market / "manifest.json", manifest)
    result = {
        "schema": SCHEMA,
        "daily_market": manifest,
        "artifacts": {"daily_market_manifest": {}},
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(tmp_path / "result.json", result)
    _atomic_json(
        tmp_path / "final_audit.json",
        {
            "status": "passed",
            "result_sha256": _sha256(tmp_path / "result.json"),
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )

    repair_artifact_paths(tmp_path)

    repaired = _read_json(tmp_path / "daily_market" / "manifest.json")
    assert repaired["artifacts"][values.name]["path"] == str(values.resolve())
    np.testing.assert_array_equal(np.load(values, allow_pickle=False), [[1.0]])
