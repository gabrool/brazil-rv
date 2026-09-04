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
    output = np.empty(array.shape, dtype=np.float32)
    output_valid = np.empty(array.shape, dtype=np.bool_)
    rank_gauss_panel_into(
        array,
        validity,
        membership,
        output,
        output_valid,
        clip=clip,
    )
    return output, output_valid


def rank_gauss_panel_into(
    values: NDArray[np.floating],
    valid: NDArray[np.bool_],
    active: NDArray[np.bool_],
    output: NDArray[np.float32],
    output_valid: NDArray[np.bool_],
    *,
    source_rows: NDArray[np.integer] | None = None,
    clip: float = 3.0,
) -> None:
    """Write row-wise rank-Gauss values into pre-allocated destinations.

    Only one cross-section (at most the number of securities) is ranked at a
    time. ``source_rows`` permits a full-history family to be normalized
    directly into a shorter store axis without creating an indexed panel copy.
    """

    array = np.asarray(values)
    validity = np.asarray(valid, dtype=np.bool_)
    membership = np.asarray(active, dtype=np.bool_)
    destination = np.asarray(output)
    destination_valid = np.asarray(output_valid)
    if array.shape != validity.shape or array.ndim < 2:
        raise ValueError("values and validity must have the same rank >= 2")
    if membership.shape != array.shape[:2]:
        raise ValueError("active must have shape [date, name]")
    if destination.dtype != np.float32 or destination_valid.dtype != np.bool_:
        raise TypeError("rank-Gauss destinations must be float32 and bool")
    if source_rows is None:
        rows = np.arange(array.shape[0], dtype=np.int64)
    else:
        raw_rows = np.asarray(source_rows)
        if raw_rows.ndim != 1 or not np.issubdtype(raw_rows.dtype, np.integer):
            raise TypeError("source_rows must be a one-dimensional integer array")
        rows = raw_rows.astype(np.int64, copy=False)
        if np.any(rows < 0) or np.any(rows >= array.shape[0]):
            raise ValueError("source_rows contains an out-of-range index")
    expected = (rows.size, *array.shape[1:])
    if destination.shape != expected or destination_valid.shape != expected:
        raise ValueError("rank-Gauss destinations are misaligned")
    trailing = int(np.prod(array.shape[2:])) if array.ndim > 2 else 1
    flat_values = array.reshape(array.shape[0], array.shape[1], trailing)
    flat_valid = validity.reshape(validity.shape[0], validity.shape[1], trailing)
    flat_output = destination.reshape(
        destination.shape[0], destination.shape[1], trailing
    )
    flat_mask = destination_valid.reshape(
        destination_valid.shape[0], destination_valid.shape[1], trailing
    )
    for output_date_index, source_date_index in enumerate(rows):
        for feature_index in range(trailing):
            transformed, mask = rank_gauss(
                flat_values[source_date_index, :, feature_index],
                flat_valid[source_date_index, :, feature_index],
                membership[source_date_index],
                clip=clip,
            )
            flat_output[output_date_index, :, feature_index] = transformed
            flat_mask[output_date_index, :, feature_index] = mask
