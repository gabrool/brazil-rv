from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from brazil_rv.modeling.contract import (
    EFFECTIVE_BATCH_SIZE,
    RECENCY_HALF_LIVES,
    ROLLING_WINDOW_DATES,
    RuntimeSettings,
)
from brazil_rv.modeling.data import (
    DateStratifiedBatchSampler,
    prepare_training_rows,
)


def _rows(date_count: int = 716) -> pl.DataFrame:
    dates = [date(2021, 8, 16) + timedelta(days=index) for index in range(date_count)]
    return pl.DataFrame(
        {
            "sample_id": np.arange(date_count, dtype=np.int64),
            "date_idx": np.arange(date_count, dtype=np.int32),
            "trade_date": dates,
            "decision_idx": np.zeros(date_count, dtype=np.int16),
        }
    )


def test_uniform_policy_is_exactly_unit_weighted() -> None:
    rows, weights, metadata = prepare_training_rows(_rows(), "uniform")
    assert rows.height == 716
    np.testing.assert_array_equal(weights, np.ones(716, dtype=np.float32))
    assert metadata["weight_mean"] == 1.0


def test_exponential_half_lives_and_mean_one_normalization() -> None:
    for policy, half_life in RECENCY_HALF_LIVES.items():
        rows, weights, metadata = prepare_training_rows(_rows(), policy)
        used = weights[rows["date_idx"].unique().sort().to_numpy()]
        assert np.isclose(used.mean(), 1.0)
        assert np.isclose(used[-1] / used[-1 - half_life], 2.0, rtol=1e-6)
        assert metadata["half_life_sessions"] == half_life


def test_rolling_policy_retains_exact_latest_window() -> None:
    rows, weights, metadata = prepare_training_rows(_rows(), "rolling_504")
    assert rows.height == ROLLING_WINDOW_DATES
    assert rows["date_idx"].min() == 716 - ROLLING_WINDOW_DATES
    assert rows["date_idx"].max() == 715
    assert np.all(weights[: 716 - ROLLING_WINDOW_DATES] == 0)
    assert np.all(weights[716 - ROLLING_WINDOW_DATES :] == 1)
    assert metadata["rolling_window_sessions"] == ROLLING_WINDOW_DATES


def test_date_replacement_is_used_only_for_undersized_date_sets() -> None:
    runtime = RuntimeSettings(
        effective_batch_size=EFFECTIVE_BATCH_SIZE,
        loader_batch_size=256,
        microbatch_size=256,
        evaluation_batch_size=256,
        num_workers=0,
    )
    assert not DateStratifiedBatchSampler(_rows(716), runtime, 11).replace_dates
    assert DateStratifiedBatchSampler(
        _rows(ROLLING_WINDOW_DATES), runtime, 11
    ).replace_dates


def test_date_sampling_is_epoch_deterministic() -> None:
    runtime = RuntimeSettings(
        effective_batch_size=8,
        loader_batch_size=4,
        microbatch_size=4,
        evaluation_batch_size=4,
        num_workers=0,
    )
    left = DateStratifiedBatchSampler(_rows(20), runtime, 29)
    right = DateStratifiedBatchSampler(_rows(20), runtime, 29)
    left.set_epoch(3)
    right.set_epoch(3)
    assert list(left) == list(right)


def test_feature_loader_to_tcn_backward_fixture(tmp_path) -> None:
    import torch

    from brazil_rv.modeling.contract import (
        GLOBAL_CONTEXT_COUNT,
        HORIZON_COUNT,
        LOCAL_CONTEXT_COUNT,
        SLOW_FEATURE_COUNT,
    )
    from brazil_rv.modeling.data import (
        BatchRequest,
        FEATURE_ARRAY_FILES,
        VectorizedFeatureDataset,
        tensorize_vectorized_batch,
    )
    from brazil_rv.modeling.engine import soft_spearman_loss
    from brazil_rv.modeling.model import build_model

    shapes = {
        "equity_features.npy": (1, 158, 405, 26),
        "equity_slow.npy": (1, 158, SLOW_FEATURE_COUNT),
        "equity_membership.npy": (1, 158),
        "equity_data_ready.npy": (1, 158),
        "context_features.npy": (1, LOCAL_CONTEXT_COUNT, 465, 26),
        "context_slow.npy": (1, LOCAL_CONTEXT_COUNT, SLOW_FEATURE_COUNT),
        "context_data_ready.npy": (1, LOCAL_CONTEXT_COUNT),
        "targets.npy": (1, 158, 55, HORIZON_COUNT),
        "global_features.npy": (1, GLOBAL_CONTEXT_COUNT, 615, 26),
        "global_slow.npy": (1, GLOBAL_CONTEXT_COUNT, 55, SLOW_FEATURE_COUNT),
        "global_data_ready.npy": (1, GLOBAL_CONTEXT_COUNT, 55),
        "label_mask.npy": (1, 158, 55, HORIZON_COUNT),
        "raw_returns.npy": (1, 158, 55, HORIZON_COUNT),
    }
    boolean = {
        "equity_membership.npy",
        "equity_data_ready.npy",
        "context_data_ready.npy",
        "global_data_ready.npy",
        "label_mask.npy",
    }
    assert set(shapes) == set(FEATURE_ARRAY_FILES)
    for name, shape in shapes.items():
        values = np.ones(shape, dtype=bool if name in boolean else np.float32)
        if name == "targets.npy":
            rank = np.linspace(-1, 1, 158, dtype=np.float32)
            values[:] = rank[None, :, None, None]
        np.save(tmp_path / name, values, allow_pickle=False)

    rows = pl.DataFrame(
        {
            "sample_id": [0],
            "date_idx": [0],
            "trade_date": [date(2024, 6, 28)],
            "decision_idx": [0],
            "equity_cutoff_index": [15],
            "context_cutoff_index": [75],
        }
    )
    dataset = VectorizedFeatureDataset(tmp_path, rows, np.ones(1, dtype=np.float32))
    batch = tensorize_vectorized_batch(dataset[BatchRequest((0,), 1)])
    model = build_model()
    predictions = model(
        batch["patches"],
        batch["history_patch_mask"],
        batch["instrument_mask"],
        batch["slow_features"],
        batch["state_position"],
    )
    assert predictions.shape == (1, 158, HORIZON_COUNT)
    loss = soft_spearman_loss(
        predictions, batch["targets"], batch["label_mask"], batch["training_weight"]
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
