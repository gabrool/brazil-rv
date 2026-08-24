from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import torch

from brazil_rv.modeling.contract import RuntimeSettings, TCN_ARCHITECTURE
from brazil_rv.modeling.data import DateStratifiedBatchSampler
from brazil_rv.modeling.ensemble_science import _bagged_dates, _derived_seed
from brazil_rv.modeling.metrics import combine_rank_predictions
from brazil_rv.modeling.model import SharedCausalTCN


def test_bagged_dates_are_deterministic_and_preserve_length() -> None:
    dates = tuple(date(2023, 1, 1) + timedelta(days=index) for index in range(47))
    first = _bagged_dates(dates, 123)
    second = _bagged_dates(dates, 123)
    assert first == second
    assert len(first) == len(dates)
    assert set(first).issubset(dates)
    assert _derived_seed("e2a", "fold_a", 11) == _derived_seed("e2a", "fold_a", 11)


def test_date_sampler_accepts_exact_multiset_with_multiplicity() -> None:
    dates = [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)]
    rows = pl.DataFrame(
        {
            "trade_date": dates,
            "decision_idx": [0, 0, 0],
        }
    )
    runtime = RuntimeSettings(
        effective_batch_size=4,
        loader_batch_size=2,
        microbatch_size=2,
        evaluation_batch_size=2,
        num_workers=0,
    )
    sampler = DateStratifiedBatchSampler(
        rows,
        runtime,
        seed=17,
        date_multiset=(dates[0], dates[0], dates[2]),
    )
    assert sampler.dates == (dates[0], dates[0], dates[2])
    assert dates[1] not in sampler.dates
    assert len(list(sampler)) == len(sampler)


def test_single_horizon_model_has_one_head_and_zero_other_outputs() -> None:
    model = SharedCausalTCN(single_horizon_index=1)
    assert model.prediction_head.out_features == 1
    predictions = torch.ones((2, 3, 1))
    mask = torch.nn.functional.one_hot(
        torch.tensor(model.single_horizon_index),
        num_classes=TCN_ARCHITECTURE.output_horizons,
    )
    expanded = predictions * mask
    assert expanded.shape == (2, 3, 3)
    assert torch.count_nonzero(expanded[..., 0]) == 0
    assert torch.count_nonzero(expanded[..., 2]) == 0


def test_rank_combiner_respects_horizon_coverage_and_fixed_weights() -> None:
    mask = np.ones((1, 4, 3), dtype=bool)
    comparator = np.asarray(
        [[[1, 4, 1], [2, 3, 2], [3, 2, 3], [4, 1, 4]]], dtype=np.float32
    )
    specialist = np.asarray(
        [[[0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0]]], dtype=np.float32
    )
    result = combine_rank_predictions(
        [comparator, specialist],
        mask,
        weights=[0.5, 0.5],
        horizon_coverage=[(0, 1, 2), (1,)],
    )
    np.testing.assert_array_equal(result[..., 0], np.asarray([[0, 1, 2, 3]]))
    np.testing.assert_array_equal(result[..., 2], np.asarray([[0, 1, 2, 3]]))
    np.testing.assert_allclose(result[..., 1], 1.5)
