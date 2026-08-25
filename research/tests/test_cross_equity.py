from __future__ import annotations

import numpy as np

from brazil_rv.modeling.cross_equity import _beta_components, _discrete_components
from brazil_rv.preprocessing.cross_equity import _peer_stat, average_linkage_clusters


def test_average_linkage_merges_small_clusters_deterministically() -> None:
    correlations = np.full((12, 12), 0.05, dtype=np.float64)
    np.fill_diagonal(correlations, 1.0)
    for start in (0, 4, 8):
        correlations[start : start + 4, start : start + 4] = 0.8
        np.fill_diagonal(correlations[start : start + 4, start : start + 4], 1.0)
    labels = average_linkage_clusters(
        correlations,
        np.ones(12, dtype=bool),
        cluster_count=6,
        minimum_size=3,
    )
    assert np.all(labels >= 0)
    assert np.all(np.bincount(labels) >= 3)
    np.testing.assert_array_equal(
        labels,
        average_linkage_clusters(
            correlations,
            np.ones(12, dtype=bool),
            cluster_count=6,
            minimum_size=3,
        ),
    )


def test_peer_stat_requires_three_valid_peers_and_excludes_self() -> None:
    measurement = np.array([10.0, 1.0, 2.0, 3.0, 100.0])
    valid = np.array([True, True, True, True, False])
    peers = np.array(
        [
            [1, 2, 3, 4],
            [0, 2, 3, -1],
            [0, 1, 4, -1],
            [0, 1, 2, -1],
            [0, 1, 2, 3],
        ]
    )
    mean, usable = _peer_stat(measurement, valid, peers, "mean")
    assert mean[0] == 2.0
    assert usable[0]
    assert usable[1]
    assert not usable[2]
    assert mean[4] == 4.0


def test_neutralization_components_are_exact_and_masked() -> None:
    scores = np.array([[[1.0], [3.0], [5.0], [9.0]]], dtype=np.float32)
    mask = np.ones_like(scores, dtype=bool)
    mask[0, 3, 0] = False
    groups = np.array([[0, 0, 1, 1]], dtype=np.int16)
    component, within = _discrete_components(scores, mask, groups)
    np.testing.assert_allclose(component[0, :3, 0], [2.0, 2.0, 5.0])
    np.testing.assert_allclose(within[0, :3, 0], [-1.0, 1.0, 0.0])
    assert component[0, 3, 0] == within[0, 3, 0] == 0

    beta_scores = np.array([[[1.0], [3.0], [5.0], [7.0]]], dtype=np.float32)
    beta = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
    beta_component, beta_within = _beta_components(
        beta_scores, np.ones_like(beta_scores, dtype=bool), beta
    )
    np.testing.assert_allclose(beta_component, beta_scores, atol=1e-6)
    np.testing.assert_allclose(beta_within, 0.0, atol=1e-6)
