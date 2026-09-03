from __future__ import annotations

from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.special import ndtri as _ndtri
except ModuleNotFoundError:  # pragma: no cover - scipy is present in the research env
    _ndtri = None


def average_ranks(values: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return stable, zero-based average ranks with exact tie handling."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("average_ranks expects one finite vector")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def midrank_unit_interval(values: NDArray[np.floating]) -> NDArray[np.float32]:
    """Scale tie-aware ranks to the closed unit interval.

    A singleton receives 0.5. For larger cross-sections the lowest and highest
    observations receive zero and one respectively.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.empty(0, dtype=np.float32)
    if array.size == 1:
        return np.full(1, 0.5, dtype=np.float32)
    return (average_ranks(array) / (array.size - 1)).astype(np.float32)


def rank_gauss(
    values: NDArray[np.floating],
    valid: NDArray[np.bool_] | None = None,
    active: NDArray[np.bool_] | None = None,
    *,
    clip: float = 3.0,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Cross-sectionally rank-Gauss one vector over valid active names.

    The specification's ``(rank - .5) / n`` uses one-based ranks. Since
    :func:`average_ranks` is zero-based, the equivalent plotting position is
    ``(average_ranks + .5) / n``. Invalid outputs are exactly zero.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("rank_gauss expects one cross-section")
    mask = np.isfinite(array)
    if valid is not None:
        candidate = np.asarray(valid, dtype=np.bool_)
        if candidate.shape != array.shape:
            raise ValueError("valid mask is misaligned")
        mask &= candidate
    if active is not None:
        candidate = np.asarray(active, dtype=np.bool_)
        if candidate.shape != array.shape:
            raise ValueError("active mask is misaligned")
        mask &= candidate
    output = np.zeros(array.shape, dtype=np.float32)
    count = int(mask.sum())
    if count:
        probabilities = (average_ranks(array[mask]) + 0.5) / count
        if _ndtri is not None:
            transformed = _ndtri(probabilities)
        else:
            normal = NormalDist()
            transformed = np.fromiter(
                (normal.inv_cdf(float(value)) for value in probabilities),
                dtype=np.float64,
                count=count,
            )
        output[mask] = np.clip(transformed, -clip, clip).astype(np.float32)
    return output, mask


def rank_gauss_panel(
    values: NDArray[np.floating],
    valid: NDArray[np.bool_],
    active: NDArray[np.bool_],
    *,
    clip: float = 3.0,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Rank-Gauss ``[date, name, ...feature]`` arrays independently by field."""

    array = np.asarray(values)
    validity = np.asarray(valid, dtype=np.bool_)
    membership = np.asarray(active, dtype=np.bool_)
    if array.shape != validity.shape or array.ndim < 2:
        raise ValueError("values and validity must have the same rank >= 2")
    if membership.shape != array.shape[:2]:
        raise ValueError("active must have shape [date, name]")
    output = np.zeros(array.shape, dtype=np.float32)
    output_valid = np.zeros(array.shape, dtype=np.bool_)
    trailing = int(np.prod(array.shape[2:])) if array.ndim > 2 else 1
    flat_values = array.reshape(array.shape[0], array.shape[1], trailing)
    flat_valid = validity.reshape(validity.shape[0], validity.shape[1], trailing)
    flat_output = output.reshape(output.shape[0], output.shape[1], trailing)
    flat_mask = output_valid.reshape(output_valid.shape[0], output_valid.shape[1], trailing)
    for date_index in range(array.shape[0]):
        for feature_index in range(trailing):
            transformed, mask = rank_gauss(
                flat_values[date_index, :, feature_index],
                flat_valid[date_index, :, feature_index],
                membership[date_index],
                clip=clip,
            )
            flat_output[date_index, :, feature_index] = transformed
            flat_mask[date_index, :, feature_index] = mask
    return output, output_valid
