import numpy as np

from brazil_rv.v2.normalization import average_ranks, rank_gauss, rank_gauss_panel


def test_average_ranks_and_rank_gauss_ties_masks_and_symmetry() -> None:
    np.testing.assert_array_equal(average_ranks(np.array([2.0, 1.0, 2.0])), [1.5, 0.0, 1.5])
    values, valid = rank_gauss(
        np.array([1.0, 2.0, 3.0, 999.0]),
        np.array([True, True, True, False]),
    )
    assert valid.tolist() == [True, True, True, False]
    assert values[3] == 0.0
    assert abs(float(values[:3].mean())) < 1e-7
    assert np.max(np.abs(values)) <= 3.0


def test_rank_gauss_panel_excludes_inactive_names() -> None:
    source = np.arange(12, dtype=float).reshape(2, 3, 2)
    valid = np.ones_like(source, dtype=bool)
    active = np.array([[True, True, False], [True, False, True]])
    values, mask = rank_gauss_panel(source, valid, active)
    assert not mask[0, 2].any()
    assert not mask[1, 1].any()
    assert np.all(values[~mask] == 0)
