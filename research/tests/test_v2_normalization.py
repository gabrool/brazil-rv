import numpy as np

from brazil_rv.v2.normalization import (
    average_ranks,
    rank_gauss,
    rank_gauss_panel,
    rank_gauss_panel_into,
)


def test_average_ranks_and_rank_gauss_ties_masks_and_symmetry() -> None:
    np.testing.assert_array_equal(
        average_ranks(np.array([2.0, 1.0, 2.0])), [1.5, 0.0, 1.5]
    )
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


def test_rank_gauss_panel_into_selects_rows_without_panel_argsort() -> None:
    source = np.arange(30, dtype=np.float32).reshape(5, 3, 2)
    valid = np.ones_like(source, dtype=np.bool_)
    active = np.ones(source.shape[:2], dtype=np.bool_)
    values = np.empty((2, 3, 2), dtype=np.float32)
    mask = np.empty_like(values, dtype=np.bool_)
    rank_gauss_panel_into(
        source,
        valid,
        active,
        values,
        mask,
        source_rows=np.asarray([1, 4]),
    )
    expected, expected_mask = rank_gauss_panel(
        source[[1, 4]], valid[[1, 4]], active[[1, 4]]
    )
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(mask, expected_mask)
