from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from brazil_rv.modeling.contract import (
    ABSOLUTE_PATCH_COUNT,
    CONTEXT_COUNT,
    DYNAMIC_CHANNEL_COUNT,
    EFFECTIVE_BATCH_SIZE,
    EQUITY_ABSOLUTE_START_PATCH,
    EQUITY_COUNT,
    GH200_RUNTIME,
    HORIZON_COUNT,
    INSTRUMENT_COUNT,
    PATCH_INPUT_WIDTH,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
)
from brazil_rv.modeling.data import (
    BatchRequest,
    DateStratifiedMicrobatchSampler,
    SequentialPaddedBatchSampler,
    VectorizedFeatureDataset,
    _validate_sample_index,
    split_sample_index,
)


def _valid_sample_index() -> pl.DataFrame:
    decisions = np.tile(np.arange(55, dtype=np.int8), 2)
    cutoff_decisions = decisions.astype(np.int16)
    return pl.DataFrame(
        {
            "sample_id": np.arange(decisions.size, dtype=np.int64),
            "trade_date": [date(2023, 1, 2)] * 55 + [date(2023, 1, 3)] * 55,
            "decision_idx": decisions,
            "equity_cutoff_index": 15 + 5 * cutoff_decisions,
            "context_cutoff_index": 75 + 5 * cutoff_decisions,
        },
        schema_overrides={"trade_date": pl.Date},
    )


def test_sample_index_integrity_rejects_invalid_rows() -> None:
    valid = _valid_sample_index()
    _validate_sample_index(valid)
    invalid_rows = (
        valid.with_columns(
            pl.when(pl.int_range(pl.len()) == 1)
            .then(pl.lit(0))
            .otherwise(pl.col("sample_id"))
            .alias("sample_id")
        ),
        valid.with_columns(
            pl.when(pl.int_range(pl.len()) == 54)
            .then(pl.lit(53))
            .otherwise(pl.col("decision_idx"))
            .alias("decision_idx")
        ),
        valid.filter(
            ~(
                (pl.col("trade_date") == date(2023, 1, 2))
                & (pl.col("decision_idx") == 54)
            )
        ),
        valid.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.lit(16))
            .otherwise(pl.col("equity_cutoff_index"))
            .alias("equity_cutoff_index")
        ),
    )
    for invalid in invalid_rows:
        with pytest.raises(ValueError):
            _validate_sample_index(invalid)


def _synthetic_store(path: Path) -> tuple[Path, pl.DataFrame]:
    date_count = 3
    generator = np.random.default_rng(17)
    equity = generator.normal(
        size=(date_count, EQUITY_COUNT, 405, DYNAMIC_CHANNEL_COUNT)
    ).astype(np.float32)
    context = generator.normal(
        size=(date_count, CONTEXT_COUNT, 465, DYNAMIC_CHANNEL_COUNT)
    ).astype(np.float32)
    equity[0, 0, 5:10] = 0.0
    membership = np.zeros((date_count, EQUITY_COUNT), dtype=bool)
    ready = np.zeros_like(membership)
    membership[:, :2] = True
    ready[:, :2] = True
    targets = generator.normal(
        size=(date_count, EQUITY_COUNT, 55, HORIZON_COUNT)
    ).astype(np.float32)
    label_mask = np.zeros_like(targets, dtype=bool)
    label_mask[:, :40] = True
    arrays = {
        "equity_features.npy": equity,
        "equity_slow.npy": generator.normal(size=(date_count, EQUITY_COUNT, 1)).astype(
            np.float32
        ),
        "equity_membership.npy": membership,
        "equity_data_ready.npy": ready,
        "context_features.npy": context,
        "context_slow.npy": generator.normal(
            size=(date_count, CONTEXT_COUNT, 3)
        ).astype(np.float32),
        "targets.npy": targets,
        "label_mask.npy": label_mask,
        "raw_returns.npy": generator.normal(
            size=(date_count, EQUITY_COUNT, 55, HORIZON_COUNT)
        ).astype(np.float32),
    }
    for filename, array in arrays.items():
        np.save(path / filename, array)
    rows = pl.DataFrame(
        {
            "sample_id": [10, 11, 12],
            "date_idx": [0, 1, 2],
            "decision_idx": [0, 2, 54],
            "equity_cutoff_index": [15, 25, 285],
            "context_cutoff_index": [75, 85, 345],
        }
    )
    return path, rows


def _direct_reference(
    store: Path, rows: pl.DataFrame, request: BatchRequest
) -> dict[str, np.ndarray]:
    arrays = {
        filename: np.load(store / filename, mmap_mode="r")
        for filename in (
            "equity_features.npy",
            "equity_slow.npy",
            "equity_membership.npy",
            "equity_data_ready.npy",
            "context_features.npy",
            "context_slow.npy",
            "targets.npy",
            "label_mask.npy",
            "raw_returns.npy",
        )
    }
    batch_size = len(request.indices)
    output = {
        "patches": np.zeros(
            (batch_size, INSTRUMENT_COUNT, ABSOLUTE_PATCH_COUNT, PATCH_INPUT_WIDTH),
            dtype=np.float32,
        ),
        "history_patch_mask": np.zeros(
            (batch_size, INSTRUMENT_COUNT, ABSOLUTE_PATCH_COUNT), dtype=bool
        ),
        "instrument_mask": np.ones((batch_size, INSTRUMENT_COUNT), dtype=bool),
        "slow_features": np.zeros((batch_size, INSTRUMENT_COUNT, 3), dtype=np.float32),
        "state_position": np.empty(batch_size, dtype=np.int64),
        "targets": np.zeros(
            (batch_size, EQUITY_COUNT, HORIZON_COUNT), dtype=np.float32
        ),
        "label_mask": np.zeros((batch_size, EQUITY_COUNT, HORIZON_COUNT), dtype=bool),
        "raw_returns": np.zeros(
            (batch_size, EQUITY_COUNT, HORIZON_COUNT), dtype=np.float32
        ),
        "sample_valid_mask": np.arange(batch_size) < request.valid_count,
        "sample_id": np.empty(batch_size, dtype=np.int64),
        "date_idx": np.empty(batch_size, dtype=np.int64),
        "decision_idx": np.empty(batch_size, dtype=np.int64),
    }
    for batch_position, row_position in enumerate(request.indices):
        row = rows.row(row_position, named=True)
        date_idx = int(row["date_idx"])
        decision_idx = int(row["decision_idx"])
        equity_cutoff = int(row["equity_cutoff_index"])
        context_cutoff = int(row["context_cutoff_index"])
        state_position = context_cutoff // 5
        equity_patch_count = equity_cutoff // 5
        active = np.asarray(
            arrays["equity_membership.npy"][date_idx]
            & arrays["equity_data_ready.npy"][date_idx]
        )
        output["instrument_mask"][batch_position, :EQUITY_COUNT] = active
        equity_prefix = np.asarray(
            arrays["equity_features.npy"][date_idx, :, :equity_cutoff]
        ).reshape(EQUITY_COUNT, equity_patch_count, PATCH_INPUT_WIDTH)
        output["patches"][
            batch_position,
            :EQUITY_COUNT,
            EQUITY_ABSOLUTE_START_PATCH:state_position,
        ] = equity_prefix * active[:, None, None]
        output["history_patch_mask"][
            batch_position,
            :EQUITY_COUNT,
            EQUITY_ABSOLUTE_START_PATCH:state_position,
        ] = active[:, None]
        output["patches"][batch_position, EQUITY_COUNT:, :state_position] = np.asarray(
            arrays["context_features.npy"][date_idx, :, :context_cutoff]
        ).reshape(CONTEXT_COUNT, state_position, PATCH_INPUT_WIDTH)
        output["history_patch_mask"][batch_position, EQUITY_COUNT:, :state_position] = (
            True
        )
        output["slow_features"][batch_position, :EQUITY_COUNT, 0] = arrays[
            "equity_slow.npy"
        ][date_idx, :, 0]
        output["slow_features"][batch_position, EQUITY_COUNT:] = arrays[
            "context_slow.npy"
        ][date_idx]
        output["state_position"][batch_position] = state_position
        output["targets"][batch_position] = arrays["targets.npy"][
            date_idx, :, decision_idx
        ]
        output["label_mask"][batch_position] = arrays["label_mask.npy"][
            date_idx, :, decision_idx
        ]
        output["raw_returns"][batch_position] = arrays["raw_returns.npy"][
            date_idx, :, decision_idx
        ]
        output["sample_id"][batch_position] = int(row["sample_id"])
        output["date_idx"][batch_position] = date_idx
        output["decision_idx"][batch_position] = decision_idx
    padded = ~output["sample_valid_mask"]
    output["targets"][padded] = 0.0
    output["label_mask"][padded] = False
    output["raw_returns"][padded] = 0.0
    output["sample_id"][padded] = -1
    output["date_idx"][padded] = -1
    output["decision_idx"][padded] = -1
    return output


def test_vectorized_batch_equivalence_and_masks(tmp_path: Path) -> None:
    store, rows = _synthetic_store(tmp_path)
    request = BatchRequest(indices=(0, 1), valid_count=2)
    actual = VectorizedFeatureDataset(store, rows)[request]
    expected = _direct_reference(store, rows, request)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    equity_source = np.load(store / "equity_features.npy", mmap_mode="r")
    context_source = np.load(store / "context_features.npy", mmap_mode="r")
    np.testing.assert_array_equal(
        actual["patches"][0, 0, EQUITY_ABSOLUTE_START_PATCH],
        equity_source[0, 0, :5].reshape(PATCH_INPUT_WIDTH),
    )
    np.testing.assert_array_equal(
        actual["patches"][0, EQUITY_COUNT, 0],
        context_source[0, 0, :5].reshape(PATCH_INPUT_WIDTH),
    )
    assert actual["history_patch_mask"][0, 0, 14]
    assert not actual["history_patch_mask"][0, 0, 15]
    assert actual["history_patch_mask"][0, 0, 13]
    assert not actual["patches"][0, 0, 13].any()
    assert not actual["instrument_mask"][0, 2]
    assert not actual["patches"][0, 2].any()
    assert actual["state_position"].tolist() == [15, 17]


def test_vectorized_future_prefix_isolation(tmp_path: Path) -> None:
    store, rows = _synthetic_store(tmp_path)
    request = BatchRequest(indices=(0, 1, 2), valid_count=3)
    baseline = VectorizedFeatureDataset(store, rows)[request]
    equity = np.load(store / "equity_features.npy", mmap_mode="r+")
    context = np.load(store / "context_features.npy", mmap_mode="r+")
    for row in rows.iter_rows(named=True):
        equity[int(row["date_idx"]), :, int(row["equity_cutoff_index"]) :] = 1e6
        context[int(row["date_idx"]), :, int(row["context_cutoff_index"]) :] = -1e6
    equity.flush()
    context.flush()
    changed = VectorizedFeatureDataset(store, rows)[request]
    for key in baseline:
        np.testing.assert_array_equal(changed[key], baseline[key])


def _sampler_rows() -> pl.DataFrame:
    first_date = date(2023, 1, 2)
    trade_dates = [
        first_date + timedelta(days=date_offset)
        for date_offset in range(571)
        for _ in range(55)
    ]
    return pl.DataFrame(
        {"sample_id": np.arange(len(trade_dates)), "trade_date": trade_dates},
        schema_overrides={"trade_date": pl.Date},
    )


def test_date_stratified_effective_batches_are_distinct_and_deterministic() -> None:
    rows = _sampler_rows()
    source_dates = rows.get_column("trade_date").to_list()
    sampler = DateStratifiedMicrobatchSampler(rows, GH200_RUNTIME, seed=11)
    sampler.set_epoch(3)
    requests = list(sampler)
    repeated = DateStratifiedMicrobatchSampler(rows, GH200_RUNTIME, seed=11)
    repeated.set_epoch(3)
    assert requests == list(repeated)
    assert len(requests) == 496
    for start in range(0, len(requests), GH200_RUNTIME.accumulation_steps):
        group = requests[start : start + GH200_RUNTIME.accumulation_steps]
        assert len(group) == GH200_RUNTIME.accumulation_steps
        assert all(
            len(request.indices) == GH200_RUNTIME.microbatch_size for request in group
        )
        positions = [index for request in group for index in request.indices]
        assert len(positions) == EFFECTIVE_BATCH_SIZE
        assert len({source_dates[index] for index in positions}) == 512


def test_fixed_evaluation_padding_invalidates_only_padded_rows(
    tmp_path: Path,
) -> None:
    store, base_rows = _synthetic_store(tmp_path)
    rows = pl.concat([base_rows] * 86, rechunk=True).with_columns(
        pl.int_range(pl.len()).alias("sample_id")
    )
    sampler = SequentialPaddedBatchSampler(
        rows.height, GH200_RUNTIME.evaluation_batch_size
    )
    requests = list(sampler)
    assert all(
        len(request.indices) == GH200_RUNTIME.evaluation_batch_size
        for request in requests
    )
    assert all(
        request.valid_count == GH200_RUNTIME.evaluation_batch_size
        for request in requests[:-1]
    )
    assert requests[-1].valid_count == 2
    batch = VectorizedFeatureDataset(store, rows)[requests[-1]]
    assert batch["sample_valid_mask"].sum() == 2
    assert not batch["sample_valid_mask"][2:].any()
    assert not batch["label_mask"][2:].any()
    assert not batch["targets"][2:].any()
    assert not batch["raw_returns"][2:].any()
    assert np.all(batch["sample_id"][2:] == -1)
    assert np.all(batch["date_idx"][2:] == -1)
    assert np.all(batch["decision_idx"][2:] == -1)


def test_gh200_batch_contract() -> None:
    assert GH200_RUNTIME.microbatch_size == 64
    assert GH200_RUNTIME.accumulation_steps == 8
    assert (
        GH200_RUNTIME.microbatch_size * GH200_RUNTIME.accumulation_steps
        == EFFECTIVE_BATCH_SIZE
    )


def test_split_and_embargo_are_disjoint() -> None:
    dates = (
        TRAIN_START,
        TRAIN_END,
        TRAIN_END + timedelta(days=3),
        VALIDATION_START,
        VALIDATION_END,
        VALIDATION_END + timedelta(days=3),
        TEST_START,
    )
    rows = pl.DataFrame(
        {"sample_id": range(len(dates)), "trade_date": dates},
        schema_overrides={"trade_date": pl.Date},
    )
    splits = split_sample_index(rows)
    ids = {
        name: set(frame.get_column("sample_id").to_list())
        for name, frame in splits.items()
    }
    names = tuple(ids)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            assert ids[left].isdisjoint(ids[right])
    assert set(splits["train"]["trade_date"]) == {TRAIN_START, TRAIN_END}
    assert set(splits["validation"]["trade_date"]) == {
        VALIDATION_START,
        VALIDATION_END,
    }
    assert set(splits["test"]["trade_date"]) == {TEST_START}
