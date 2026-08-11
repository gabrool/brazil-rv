from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from brazil_rv.modeling.contract import (
    ABSOLUTE_PATCH_COUNT,
    DECISION_GLOBAL_INDICES,
    EXPECTED_SPLIT_DATE_COUNTS,
    EXPECTED_SPLIT_SAMPLE_COUNTS,
    GLOBAL_CONTEXT_COUNT,
    GLOBAL_WINDOW_MINUTES,
    LOCAL_CONTEXT_COUNT,
    DYNAMIC_CHANNEL_COUNT,
    EFFECTIVE_BATCH_SIZE,
    EQUITY_ABSOLUTE_START_PATCH,
    EQUITY_COUNT,
    GH200_RUNTIME,
    HORIZON_COUNT,
    INSTRUMENT_COUNT,
    PATCH_INPUT_WIDTH,
    SLOW_FEATURE_COUNT,
    TABULAR_FEATURE_COUNT,
    TABULAR_OFFSETS,
    TEST_START,
    TCNSettings,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    resolve_tcn_architecture,
)
from brazil_rv.modeling.data import (
    BatchRequest,
    DateStratifiedMicrobatchSampler,
    FEATURE_ARRAY_FILES,
    SequentialPaddedBatchSampler,
    TabularRowIterator,
    VectorizedFeatureDataset,
    _validate_sample_index,
    split_sample_index,
)
from brazil_rv.preprocessing.peer_features import validate_peer_arrays


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
            pl.when(pl.int_range(pl.len()) == valid.height - 1)
            .then(pl.lit(valid.height))
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
        size=(date_count, LOCAL_CONTEXT_COUNT, 465, DYNAMIC_CHANNEL_COUNT)
    ).astype(np.float32)
    global_features = generator.normal(
        size=(date_count, GLOBAL_CONTEXT_COUNT, 615, DYNAMIC_CHANNEL_COUNT)
    ).astype(np.float32)
    global_features[..., 5] = 1.0
    global_ready = np.ones((date_count, GLOBAL_CONTEXT_COUNT, 55), dtype=bool)
    global_ready[1, 2, 2] = False
    equity[0, 0, 5:10] = 0.0
    membership = np.zeros((date_count, EQUITY_COUNT), dtype=bool)
    ready = np.zeros_like(membership)
    membership[:, :2] = True
    ready[:, :2] = True
    targets = generator.normal(
        size=(date_count, EQUITY_COUNT, 55, HORIZON_COUNT)
    ).astype(np.float32)
    label_mask = np.zeros_like(targets, dtype=bool)
    label_mask[:, :2] = True
    arrays = {
        "equity_features.npy": equity,
        "equity_slow.npy": generator.normal(
            size=(date_count, EQUITY_COUNT, SLOW_FEATURE_COUNT)
        ).astype(np.float32),
        "equity_membership.npy": membership,
        "equity_data_ready.npy": ready,
        "equity_peer_features.npy": generator.uniform(
            -0.9, 0.9, size=(date_count, EQUITY_COUNT, 405, 6)
        ).astype(np.float32),
        "equity_peer_valid.npy": np.ones(
            (date_count, EQUITY_COUNT, 405, 4), dtype=bool
        ),
        "context_features.npy": context,
        "context_slow.npy": generator.normal(
            size=(date_count, LOCAL_CONTEXT_COUNT, SLOW_FEATURE_COUNT)
        ).astype(np.float32),
        "context_data_ready.npy": np.ones(
            (date_count, LOCAL_CONTEXT_COUNT), dtype=bool
        ),
        "global_features.npy": global_features,
        "global_slow.npy": generator.normal(
            size=(date_count, GLOBAL_CONTEXT_COUNT, 55, SLOW_FEATURE_COUNT)
        ).astype(np.float32),
        "global_data_ready.npy": global_ready,
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
            "context_data_ready.npy",
            "global_features.npy",
            "global_slow.npy",
            "global_data_ready.npy",
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
        "slow_features": np.zeros(
            (batch_size, INSTRUMENT_COUNT, SLOW_FEATURE_COUNT), dtype=np.float32
        ),
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
        local_slice = slice(EQUITY_COUNT, EQUITY_COUNT + LOCAL_CONTEXT_COUNT)
        local_ready = np.asarray(arrays["context_data_ready.npy"][date_idx])
        local_prefix = np.asarray(
            arrays["context_features.npy"][date_idx, :, :context_cutoff]
        ).reshape(LOCAL_CONTEXT_COUNT, state_position, PATCH_INPUT_WIDTH)
        output["patches"][batch_position, local_slice, :state_position] = (
            local_prefix * local_ready[:, None, None]
        )
        output["history_patch_mask"][batch_position, local_slice, :state_position] = (
            local_ready[:, None]
        )
        output["instrument_mask"][batch_position, local_slice] = local_ready
        output["slow_features"][batch_position, :EQUITY_COUNT] = (
            arrays["equity_slow.npy"][date_idx] * active[:, None]
        )
        output["slow_features"][batch_position, local_slice] = (
            arrays["context_slow.npy"][date_idx] * local_ready[:, None]
        )

        global_start = EQUITY_COUNT + LOCAL_CONTEXT_COUNT
        global_ready = np.asarray(
            arrays["global_data_ready.npy"][date_idx, :, decision_idx]
        )
        global_cutoff = DECISION_GLOBAL_INDICES[decision_idx]
        global_prefix = np.asarray(
            arrays["global_features.npy"][
                date_idx, :, global_cutoff - GLOBAL_WINDOW_MINUTES : global_cutoff
            ]
        ).reshape(GLOBAL_CONTEXT_COUNT, ABSOLUTE_PATCH_COUNT, PATCH_INPUT_WIDTH)
        output["patches"][batch_position, global_start:] = (
            global_prefix * global_ready[:, None, None]
        )
        output["history_patch_mask"][batch_position, global_start:] = global_ready[
            :, None
        ]
        output["instrument_mask"][batch_position, global_start:] = global_ready
        output["slow_features"][batch_position, global_start:] = (
            arrays["global_slow.npy"][date_idx, :, decision_idx] * global_ready[:, None]
        )
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
    actual = VectorizedFeatureDataset(store, rows, "context_pooled", "enabled")[request]
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


def test_current_model_batches_ignore_peer_arrays(tmp_path: Path) -> None:
    store, rows = _synthetic_store(tmp_path)
    request = BatchRequest(indices=(0, 1), valid_count=2)
    validate_peer_arrays(
        np.load(store / "equity_peer_features.npy", mmap_mode="r"),
        np.load(store / "equity_peer_valid.npy", mmap_mode="r"),
    )
    baseline = VectorizedFeatureDataset(store, rows, "context_pooled", "enabled")[
        request
    ]
    assert "equity_peer_features.npy" not in FEATURE_ARRAY_FILES
    assert "equity_peer_valid.npy" not in FEATURE_ARRAY_FILES
    peer_features = np.load(store / "equity_peer_features.npy", mmap_mode="r+")
    peer_valid = np.load(store / "equity_peer_valid.npy", mmap_mode="r+")
    peer_features[...] = 1_000_000.0
    peer_valid[...] = False
    peer_features.flush()
    peer_valid.flush()
    changed = VectorizedFeatureDataset(store, rows, "context_pooled", "enabled")[
        request
    ]
    for key in baseline:
        np.testing.assert_array_equal(changed[key], baseline[key])


def test_vectorized_future_prefix_isolation(tmp_path: Path) -> None:
    store, rows = _synthetic_store(tmp_path)
    request = BatchRequest(indices=(0, 1, 2), valid_count=3)
    baseline = VectorizedFeatureDataset(store, rows, "context_pooled", "enabled")[
        request
    ]
    equity = np.load(store / "equity_features.npy", mmap_mode="r+")
    context = np.load(store / "context_features.npy", mmap_mode="r+")
    for row in rows.iter_rows(named=True):
        equity[int(row["date_idx"]), :, int(row["equity_cutoff_index"]) :] = 1e6
        context[int(row["date_idx"]), :, int(row["context_cutoff_index"]) :] = -1e6
    equity.flush()
    context.flush()
    changed = VectorizedFeatureDataset(store, rows, "context_pooled", "enabled")[
        request
    ]
    for key in baseline:
        np.testing.assert_array_equal(changed[key], baseline[key])


def _sampler_rows() -> pl.DataFrame:
    first_date = date(2023, 1, 2)
    trade_dates = [
        first_date + timedelta(days=date_offset)
        for date_offset in range(716)
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
    assert len(requests) == 616
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
    batch = VectorizedFeatureDataset(store, rows, "context_pooled", "enabled")[
        requests[-1]
    ]
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


def test_audited_split_counts_are_exact() -> None:
    split_starts = {
        "train": TRAIN_START,
        "embargo_1": TRAIN_END + timedelta(days=1),
        "validation": VALIDATION_START,
        "embargo_2": VALIDATION_END + timedelta(days=1),
        "test": TEST_START,
    }
    trade_dates = [
        split_starts[split] + timedelta(days=offset)
        for split, count in EXPECTED_SPLIT_DATE_COUNTS.items()
        for offset in range(count)
        for _ in range(55)
    ]
    rows = pl.DataFrame(
        {"sample_id": np.arange(len(trade_dates)), "trade_date": trade_dates},
        schema_overrides={"trade_date": pl.Date},
    )
    splits = split_sample_index(rows)
    for split, expected_dates in EXPECTED_SPLIT_DATE_COUNTS.items():
        assert splits[split]["trade_date"].n_unique() == expected_dates
        assert splits[split].height == EXPECTED_SPLIT_SAMPLE_COUNTS[split]


def test_family_specific_batches_construct_only_required_inputs(tmp_path: Path) -> None:
    store, rows = _synthetic_store(tmp_path)
    request = BatchRequest(indices=(0,), valid_count=1)
    patch_batch = VectorizedFeatureDataset(store, rows, "context_pooled", "enabled")[
        request
    ]
    assert "patches" in patch_batch and "tabular_features" not in patch_batch
    mlp_batch = VectorizedFeatureDataset(store, rows, "mlp", "enabled")[request]
    assert "tabular_features" in mlp_batch and "patches" not in mlp_batch
    assert mlp_batch["tabular_features"].shape == (
        1,
        EQUITY_COUNT,
        TABULAR_FEATURE_COUNT,
    )
    pooled_batch = VectorizedFeatureDataset(store, rows, "pooled_market", None)[request]
    assert not pooled_batch["patches"][:, EQUITY_COUNT:].any()
    assert not pooled_batch["instrument_mask"][:, EQUITY_COUNT:].any()
    assert not pooled_batch["slow_features"][:, EQUITY_COUNT:].any()


@pytest.mark.parametrize(
    ("fusion", "needs_context"),
    (
        ("none", False),
        ("context_only", True),
        ("pooled_market", False),
        ("context_pooled", True),
    ),
)
def test_tcn_batches_construct_only_selected_context(
    tmp_path: Path, fusion: str, needs_context: bool
) -> None:
    store, rows = _synthetic_store(tmp_path)
    architecture = resolve_tcn_architecture(TCNSettings(fusion, 64, "short", "gelu"))
    dataset = VectorizedFeatureDataset(
        store, rows, "tcn", "enabled" if needs_context else None, architecture
    )
    batch = dataset[BatchRequest(indices=(0,), valid_count=1)]
    assert bool(batch["patches"][:, EQUITY_COUNT:].any()) is needs_context
    assert bool(batch["instrument_mask"][:, EQUITY_COUNT:].any()) is needs_context
    assert bool(batch["slow_features"][:, EQUITY_COUNT:].any()) is needs_context

    if not needs_context:
        assert dataset._arrays is not None
        assert not any(name.startswith("context_") for name in dataset._arrays)


def test_unavailable_context_is_zeroed_before_tensor_construction(
    tmp_path: Path,
) -> None:
    store, rows = _synthetic_store(tmp_path)
    context = np.load(store / "context_features.npy", mmap_mode="r+")
    context_slow = np.load(store / "context_slow.npy", mmap_mode="r+")
    context_ready = np.load(store / "context_data_ready.npy", mmap_mode="r+")
    context_ready[0, 0] = False
    context_ready.flush()

    architecture = resolve_tcn_architecture(
        TCNSettings("context_only", 64, "short", "gelu")
    )
    dataset = VectorizedFeatureDataset(store, rows, "tcn", "enabled", architecture)
    request = BatchRequest(indices=(0,), valid_count=1)
    baseline = dataset[request]
    context[0, 0] = 123.0
    context_slow[0, 0] = 456.0
    context.flush()
    context_slow.flush()
    batch = dataset[request]
    for key in baseline:
        np.testing.assert_array_equal(batch[key], baseline[key])
    unavailable = EQUITY_COUNT
    ready = EQUITY_COUNT + 1
    assert not batch["patches"][0, unavailable].any()
    assert not batch["history_patch_mask"][0, unavailable].any()
    assert not batch["slow_features"][0, unavailable].any()
    assert not batch["instrument_mask"][0, unavailable]
    assert batch["patches"][0, ready, :15].any()
    assert batch["history_patch_mask"][0, ready, :15].all()
    assert batch["slow_features"][0, ready].any()
    assert batch["instrument_mask"][0, ready]


def test_unavailable_local_context_is_masked_in_tabular_inputs(
    tmp_path: Path,
) -> None:
    store, rows = _synthetic_store(tmp_path)
    context = np.load(store / "context_features.npy", mmap_mode="r+")
    context_slow = np.load(store / "context_slow.npy", mmap_mode="r+")
    context_ready = np.load(store / "context_data_ready.npy", mmap_mode="r+")
    context_ready[0, 0] = False
    context_ready.flush()
    request = BatchRequest(indices=(0,), valid_count=1)
    dataset = VectorizedFeatureDataset(store, rows, "mlp", "enabled")
    baseline = dataset[request]["tabular_features"]

    context[0, 0] = 123.0
    context_slow[0, 0] = 456.0
    context.flush()
    context_slow.flush()
    changed = dataset[request]["tabular_features"]
    np.testing.assert_array_equal(changed, baseline)

    local_dynamic_start = (
        SLOW_FEATURE_COUNT + len(TABULAR_OFFSETS) * DYNAMIC_CHANNEL_COUNT
    )
    local_slot_dynamic_width = len(TABULAR_OFFSETS) * 16
    global_dynamic_stop = (
        local_dynamic_start
        + LOCAL_CONTEXT_COUNT * local_slot_dynamic_width
        + GLOBAL_CONTEXT_COUNT * len(TABULAR_OFFSETS) * 16
    )
    local_slow_start = global_dynamic_stop
    readiness_start = TABULAR_FEATURE_COUNT - (
        LOCAL_CONTEXT_COUNT + GLOBAL_CONTEXT_COUNT
    )
    assert not baseline[
        0, 0, local_dynamic_start : local_dynamic_start + local_slot_dynamic_width
    ].any()
    assert not baseline[
        0, 0, local_slow_start : local_slow_start + SLOW_FEATURE_COUNT
    ].any()
    np.testing.assert_array_equal(
        baseline[0, 0, readiness_start : readiness_start + LOCAL_CONTEXT_COUNT],
        context_ready[0].astype(np.float32),
    )
    assert baseline[
        0,
        0,
        local_slow_start + SLOW_FEATURE_COUNT : local_slow_start
        + 2 * SLOW_FEATURE_COUNT,
    ].any()


def test_compact_tabular_offsets_validity_and_future_causality(tmp_path: Path) -> None:
    store, rows = _synthetic_store(tmp_path)
    equity = np.load(store / "equity_features.npy", mmap_mode="r+")
    context = np.load(store / "context_features.npy", mmap_mode="r+")
    global_features = np.load(store / "global_features.npy", mmap_mode="r+")
    equity[0, 0, :, 5] = 0.0
    equity[0, 0, 14, :DYNAMIC_CHANNEL_COUNT] = np.arange(
        DYNAMIC_CHANNEL_COUNT, dtype=np.float32
    )
    equity[0, 0, 14, 5] = 1.0
    context[0, 0, :, 5] = 0.0
    for minute in (74, 59, 44, 14):
        context[0, 0, minute, :16] = minute + np.arange(16, dtype=np.float32)
        context[0, 0, minute, 5] = 1.0
    equity.flush()
    context.flush()

    request = BatchRequest(indices=(0,), valid_count=1)
    baseline = VectorizedFeatureDataset(store, rows, "mlp", "enabled")[request]
    features = baseline["tabular_features"]

    assert features.shape[-1] == TABULAR_FEATURE_COUNT
    np.testing.assert_array_equal(
        features[0, 0, 32 : 32 + DYNAMIC_CHANNEL_COUNT],
        equity[0, 0, 14],
    )
    assert not features[
        0, 0, 32 + DYNAMIC_CHANNEL_COUNT : 32 + 5 * DYNAMIC_CHANNEL_COUNT
    ].any()
    context_start = 32 + 5 * DYNAMIC_CHANNEL_COUNT
    np.testing.assert_array_equal(
        features[0, 0, context_start : context_start + 16], context[0, 0, 74, :16]
    )
    global_start = context_start + LOCAL_CONTEXT_COUNT * len(TABULAR_OFFSETS) * 16
    np.testing.assert_array_equal(
        features[0, 0, global_start : global_start + 16],
        global_features[0, 0, 344, :16],
    )
    validity_start = TABULAR_FEATURE_COUNT - (
        (1 + LOCAL_CONTEXT_COUNT + GLOBAL_CONTEXT_COUNT) * len(TABULAR_OFFSETS)
        + LOCAL_CONTEXT_COUNT
        + GLOBAL_CONTEXT_COUNT
    )
    assert features[0, 0, validity_start : validity_start + 5].tolist() == [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert features[0, 0, validity_start + 5 : validity_start + 10].tolist() == [
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
    ]
    assert not features[0, 2].any()

    equity[0, :, 15:] = 100_000.0
    context[0, :, 75:] = -100_000.0
    global_features[0, :, 345:] = 50_000.0
    equity.flush()
    context.flush()
    global_features.flush()
    changed = VectorizedFeatureDataset(store, rows, "mlp", "enabled")[request]
    np.testing.assert_array_equal(
        baseline["tabular_features"], changed["tabular_features"]
    )
    assert TABULAR_OFFSETS == (0, 15, 30, 60, 120)


def test_masked_global_context_keeps_layout_and_contributes_exact_zeros(
    tmp_path: Path,
) -> None:
    store, rows = _synthetic_store(tmp_path)
    request = BatchRequest(indices=(0,), valid_count=1)
    enabled = VectorizedFeatureDataset(store, rows, "context_pooled", "enabled")[
        request
    ]
    masked = VectorizedFeatureDataset(store, rows, "context_pooled", "masked")[request]
    assert enabled.keys() == masked.keys()
    for key in enabled:
        assert enabled[key].shape == masked[key].shape

    global_start = EQUITY_COUNT + LOCAL_CONTEXT_COUNT
    assert enabled["patches"][:, global_start:].any()
    assert enabled["instrument_mask"][:, global_start:].any()
    assert enabled["slow_features"][:, global_start:].any()
    assert not masked["patches"][:, global_start:].any()
    assert not masked["history_patch_mask"][:, global_start:].any()
    assert not masked["instrument_mask"][:, global_start:].any()
    assert not masked["slow_features"][:, global_start:].any()

    enabled_tabular = VectorizedFeatureDataset(store, rows, "mlp", "enabled")[request][
        "tabular_features"
    ]
    masked_tabular = VectorizedFeatureDataset(store, rows, "mlp", "masked")[request][
        "tabular_features"
    ]
    global_dynamic_start = (
        SLOW_FEATURE_COUNT
        + len(TABULAR_OFFSETS) * DYNAMIC_CHANNEL_COUNT
        + LOCAL_CONTEXT_COUNT * len(TABULAR_OFFSETS) * 16
    )
    global_dynamic_stop = (
        global_dynamic_start + GLOBAL_CONTEXT_COUNT * len(TABULAR_OFFSETS) * 16
    )
    global_slow_start = global_dynamic_stop + LOCAL_CONTEXT_COUNT * SLOW_FEATURE_COUNT
    global_slow_stop = global_slow_start + GLOBAL_CONTEXT_COUNT * SLOW_FEATURE_COUNT
    global_validity_start = (
        global_slow_stop
        + 2
        + len(TABULAR_OFFSETS)
        + LOCAL_CONTEXT_COUNT * len(TABULAR_OFFSETS)
    )
    global_validity_stop = global_validity_start + GLOBAL_CONTEXT_COUNT * len(
        TABULAR_OFFSETS
    )
    global_readiness_start = global_validity_stop + LOCAL_CONTEXT_COUNT
    global_columns = np.concatenate(
        (
            np.arange(global_dynamic_start, global_dynamic_stop),
            np.arange(global_slow_start, global_slow_stop),
            np.arange(global_validity_start, global_validity_stop),
            np.arange(global_readiness_start, TABULAR_FEATURE_COUNT),
        )
    )
    keep = np.ones(TABULAR_FEATURE_COUNT, dtype=bool)
    keep[global_columns] = False
    assert enabled_tabular[..., global_columns].any()
    assert not masked_tabular[..., global_columns].any()
    np.testing.assert_array_equal(enabled_tabular[..., keep], masked_tabular[..., keep])


def test_tabular_row_iterator_masks_horizons_and_equalizes_samples(
    tmp_path: Path,
) -> None:
    store, rows = _synthetic_store(tmp_path)
    label_mask = np.load(store / "label_mask.npy", mmap_mode="r+")
    label_mask[0, 1, 0, 1] = False
    label_mask.flush()
    horizon_0 = list(
        TabularRowIterator(
            store, rows, 0, device="cpu", global_context="enabled", batch_size=2
        )
    )
    horizon_1 = list(
        TabularRowIterator(
            store, rows, 1, device="cpu", global_context="enabled", batch_size=2
        )
    )
    assert sum(batch.features.shape[0] for batch in horizon_0) == 6
    assert sum(batch.features.shape[0] for batch in horizon_1) == 5
    for batches in (horizon_0, horizon_1):
        for batch in batches:
            assert batch.features.shape[1] == TABULAR_FEATURE_COUNT
            for sample_id in np.unique(batch.sample_id):
                on_sample = batch.sample_id == sample_id
                np.testing.assert_allclose(batch.weights[on_sample].sum(), 1.0)
    first_horizon_1 = horizon_1[0]
    on_first_sample = first_horizon_1.sample_id == 10
    assert on_first_sample.sum() == 1
    assert first_horizon_1.weights[on_first_sample].item() == 1.0


def test_tabular_row_iterator_cuda_materializes_only_model_inputs(
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA tensor materialization requires an available GPU")
    store, rows = _synthetic_store(tmp_path)
    batch = next(
        iter(
            TabularRowIterator(store, rows, 0, device="cuda", global_context="enabled")
        )
    )
    for value in (batch.features, batch.labels, batch.weights):
        assert isinstance(value, torch.Tensor)
        assert value.device.type == "cuda"
    for value in (
        batch.sample_id,
        batch.date_idx,
        batch.decision_idx,
        batch.equity_slot,
    ):
        assert isinstance(value, np.ndarray)
