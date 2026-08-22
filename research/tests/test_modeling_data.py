from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from brazil_rv.modeling.contract import EFFECTIVE_BATCH_SIZE, RuntimeSettings
from brazil_rv.modeling.data import (
    DateStratifiedBatchSampler,
    discovery_folds,
    third_discovery_fold,
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


def test_discovery_folds_have_exact_expanding_windows() -> None:
    fold_a, fold_b = discovery_folds(_rows())
    assert (fold_a.name, fold_b.name) == ("fold_a", "fold_b")
    assert fold_a.fit_rows["date_idx"].n_unique() == 512
    assert fold_a.selection_rows["date_idx"].n_unique() == 102
    assert fold_b.fit_rows["date_idx"].n_unique() == 614
    assert fold_b.selection_rows["date_idx"].n_unique() == 102
    assert set(fold_a.selection_rows["date_idx"]).isdisjoint(
        set(fold_b.selection_rows["date_idx"])
    )
    assert fold_a.selection_rows["date_idx"].min() == 512
    assert fold_b.selection_rows["date_idx"].min() == 614


def test_both_fold_training_windows_sample_512_distinct_dates() -> None:
    runtime = RuntimeSettings(
        effective_batch_size=EFFECTIVE_BATCH_SIZE,
        loader_batch_size=256,
        microbatch_size=256,
        evaluation_batch_size=256,
        num_workers=0,
    )
    for fold in discovery_folds(_rows()):
        sampler = DateStratifiedBatchSampler(fold.fit_rows, runtime, 11)
        assert not sampler.replace_dates
        first_two_loader_batches = list(sampler)[:2]
        positions = (
            first_two_loader_batches[0].indices + first_two_loader_batches[1].indices
        )
        assert len(positions) == len(set(positions)) == 512


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


def test_third_fold_uses_the_predeclared_calendar_and_replacement() -> None:
    fit_dates = [date(2021, 8, 16) + timedelta(days=index) for index in range(406)]
    fit_dates.append(date(2023, 3, 31))
    selection_dates = [date(2023, 4, 3) + timedelta(days=index) for index in range(104)]
    selection_dates.append(date(2023, 8, 31))
    tail_dates = [date(2023, 9, 1) + timedelta(days=index) for index in range(204)]
    dates = [*fit_dates, *selection_dates, *tail_dates]
    rows = _rows().with_columns(pl.Series("trade_date", dates, dtype=pl.Date))
    fold = third_discovery_fold(rows)
    assert fold.fit_rows["trade_date"].min() == date(2021, 8, 16)
    assert fold.fit_rows["trade_date"].max() == date(2023, 3, 31)
    assert fold.selection_rows["trade_date"].min() == date(2023, 4, 3)
    assert fold.selection_rows["trade_date"].max() == date(2023, 8, 31)
    runtime = RuntimeSettings(num_workers=0)
    assert DateStratifiedBatchSampler(fold.fit_rows, runtime, 11).replace_dates


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
    dataset = VectorizedFeatureDataset(
        tmp_path,
        rows,
        zero_dynamic_channels=(0,),
        zero_slow_fields=(1,),
    )
    batch = tensorize_vectorized_batch(dataset[BatchRequest((0,), 1)])
    assert not batch["patches"][0, :158, :, 0::26].any()
    assert batch["patches"][0, 158:, :, 0::26].any()
    assert not batch["slow_features"][0, :158, 1].any()
    assert batch["slow_features"][0, 158:, 1].any()
    assert "continuous_targets" not in batch
    assert "training_weight" not in batch
    model = build_model()
    predictions = model(
        batch["patches"],
        batch["history_patch_mask"],
        batch["instrument_mask"],
        batch["slow_features"],
        batch["state_position"],
    )
    assert predictions.shape == (1, 158, HORIZON_COUNT)
    loss = soft_spearman_loss(predictions, batch["targets"], batch["label_mask"])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
