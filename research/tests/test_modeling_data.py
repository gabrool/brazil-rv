from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from brazil_rv.modeling.contract import (
    ABSOLUTE_PATCH_COUNT,
    EFFECTIVE_BATCH_SIZE,
    EQUITY_ABSOLUTE_START_PATCH,
    EQUITY_COUNT,
    GLOBAL_CONTEXT_COUNT,
    LOCAL_CONTEXT_COUNT,
    PATCH_INPUT_WIDTH,
    RuntimeSettings,
    SLOW_FEATURE_COUNT,
    TABULAR_FEATURE_COUNT,
)
from brazil_rv.modeling.data import (
    DateStratifiedMicrobatchSampler,
    DecisionGroupedBatchSampler,
    _build_patch_batch,
    _build_peer_state,
    _validate_sample_index,
    build_tabular_batch,
    select_sample_split,
)


def _arrays() -> dict[str, np.ndarray]:
    equity_features = np.arange(EQUITY_COUNT * 405 * 26, dtype=np.float32).reshape(
        1, EQUITY_COUNT, 405, 26
    )
    context_features = np.ones((1, LOCAL_CONTEXT_COUNT, 465, 26), dtype=np.float32)
    global_features = np.ones((1, GLOBAL_CONTEXT_COUNT, 615, 26), dtype=np.float32)
    equity_features[..., 5] = 1
    context_features[..., 5] = 1
    global_features[..., 5] = 1
    return {
        "equity_features.npy": equity_features,
        "equity_slow.npy": np.ones(
            (1, EQUITY_COUNT, SLOW_FEATURE_COUNT), dtype=np.float32
        ),
        "equity_membership.npy": np.ones((1, EQUITY_COUNT), dtype=bool),
        "equity_data_ready.npy": np.ones((1, EQUITY_COUNT), dtype=bool),
        "context_features.npy": context_features,
        "context_slow.npy": np.ones(
            (1, LOCAL_CONTEXT_COUNT, SLOW_FEATURE_COUNT), dtype=np.float32
        ),
        "context_data_ready.npy": np.ones((1, LOCAL_CONTEXT_COUNT), dtype=bool),
        "global_features.npy": global_features,
        "global_slow.npy": np.ones(
            (1, GLOBAL_CONTEXT_COUNT, 55, SLOW_FEATURE_COUNT), dtype=np.float32
        ),
        "global_data_ready.npy": np.ones((1, GLOBAL_CONTEXT_COUNT, 55), dtype=bool),
        "equity_peer_features.npy": np.zeros(
            (1, EQUITY_COUNT, 405, 6), dtype=np.float32
        ),
        "equity_peer_valid.npy": np.zeros((1, EQUITY_COUNT, 405, 4), dtype=bool),
    }


def _patch(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return _build_patch_batch(
        arrays,
        np.array([0]),
        np.array([15]),
        np.array([0]),
        np.array([75]),
        np.ones((1, EQUITY_COUNT), dtype=bool),
        "enabled",
    )


def test_current_full_universe_batch_shapes_masks_and_context_policy() -> None:
    batch = _patch(_arrays())
    assert batch["patches"].shape == (
        1,
        EQUITY_COUNT + 15,
        ABSOLUTE_PATCH_COUNT,
        PATCH_INPUT_WIDTH,
    )
    assert batch["slow_features"].shape == (1, EQUITY_COUNT + 15, SLOW_FEATURE_COUNT)
    assert batch["instrument_mask"][:, :EQUITY_COUNT].all()
    assert batch["history_patch_mask"][0, 0, :EQUITY_ABSOLUTE_START_PATCH].sum() == 0
    assert batch["history_patch_mask"][0, 0, EQUITY_ABSOLUTE_START_PATCH:15].all()
    assert not batch["instrument_mask"][0, EQUITY_COUNT]  # WIN dropped
    assert batch["instrument_mask"][0, EQUITY_COUNT + 1]  # WDO retained
    globals_ = batch["instrument_mask"][0, EQUITY_COUNT + LOCAL_CONTEXT_COUNT :]
    np.testing.assert_array_equal(
        globals_, [False, False, True, True, False, False, False, False]
    )
    assert not batch["slow_features"][0, :EQUITY_COUNT, 20].any()  # beta_to_WIN


def test_entry_and_future_bars_never_enter_patch_batch() -> None:
    arrays = _arrays()
    baseline = _patch(arrays)
    arrays["equity_features.npy"][:, :, 15:] = 1e9
    changed = _patch(arrays)
    np.testing.assert_array_equal(baseline["patches"], changed["patches"])


def test_unavailable_context_is_masked_without_gating_equities() -> None:
    arrays = _arrays()
    arrays["context_data_ready.npy"][0, 2] = False
    arrays["global_data_ready.npy"][0, 2, 0] = False
    batch = _patch(arrays)
    assert batch["instrument_mask"][0, :EQUITY_COUNT].all()
    assert not batch["instrument_mask"][0, EQUITY_COUNT + 2]
    assert not batch["patches"][0, EQUITY_COUNT + 2].any()
    assert not batch["instrument_mask"][0, EQUITY_COUNT + LOCAL_CONTEXT_COUNT + 2]


def test_selected_peer_state_has_current_six_field_order_and_masks() -> None:
    arrays = _arrays()
    arrays["equity_peer_features.npy"][0, 0, 14, :4] = [1, 2, 3, 4]
    arrays["equity_peer_valid.npy"][0, 0, 14, :2] = [True, False]
    state = _build_peer_state(
        arrays, np.array([0]), np.array([15]), np.ones((1, EQUITY_COUNT), dtype=bool)
    )
    np.testing.assert_array_equal(state[0, 0], [1, 0, 3, 0, 1, 0])
    assert state.shape == (1, EQUITY_COUNT, 6)


def test_tabular_batch_uses_same_policy_and_width() -> None:
    arrays = _arrays()
    batch = build_tabular_batch(
        arrays,
        np.array([0]),
        np.array([0]),
        np.array([15]),
        np.array([75]),
        np.ones((1, EQUITY_COUNT), dtype=bool),
        "enabled",
    )
    assert batch["tabular_features"].shape == (1, EQUITY_COUNT, TABULAR_FEATURE_COUNT)
    assert not batch["tabular_features"][..., 20].any()
    assert np.isfinite(batch["tabular_features"]).all()


def _sample_index(days: int, decisions: int = 55) -> pl.DataFrame:
    rows = []
    sample_id = 0
    for day in range(days):
        trade_date = date(2022, 1, 1) + timedelta(days=day)
        for decision in range(decisions):
            rows.append(
                {
                    "sample_id": sample_id,
                    "trade_date": trade_date,
                    "date_idx": day,
                    "decision_idx": decision,
                    "equity_cutoff_index": 15 + 5 * decision,
                    "context_cutoff_index": 75 + 5 * decision,
                }
            )
            sample_id += 1
    return pl.DataFrame(rows)


def test_sample_index_enforces_complete_causal_decision_grid() -> None:
    rows = _sample_index(2)
    _validate_sample_index(rows)
    with pytest.raises(ValueError, match="decisions"):
        _validate_sample_index(
            rows.filter(pl.col("sample_id") != 3)
            .drop("sample_id")
            .with_row_index("sample_id")
        )
    corrupt = rows.with_columns(
        pl.when(pl.col("sample_id") == 0)
        .then(16)
        .otherwise(pl.col("equity_cutoff_index"))
        .alias("equity_cutoff_index")
    )
    with pytest.raises(ValueError, match="causal"):
        _validate_sample_index(corrupt)


def test_date_stratified_sampler_uses_full_grid_distinct_dates_and_decision_order() -> (
    None
):
    rows = _sample_index(EFFECTIVE_BATCH_SIZE)
    runtime = RuntimeSettings(
        microbatch_size=64,
        accumulation_steps=8,
        evaluation_batch_size=32,
        num_workers=0,
    )
    sampler = DateStratifiedMicrobatchSampler(rows, runtime, seed=29)
    requests = [next(iter(sampler))]
    # Reiterate once and inspect the complete first effective batch.
    iterator = iter(sampler)
    requests = [next(iterator) for _ in range(runtime.accumulation_steps)]
    positions = [position for request in requests for position in request.indices]
    selected = rows[positions]
    assert selected.get_column("trade_date").n_unique() == EFFECTIVE_BATCH_SIZE
    decisions = selected.get_column("decision_idx").to_list()
    assert decisions == sorted(decisions)
    assert min(decisions) >= 0 and max(decisions) < 55


def test_padding_is_explicit_and_does_not_create_real_rows() -> None:
    rows = pl.DataFrame({"sample_id": [4, 2, 3, 1, 0], "decision_idx": [1, 0, 1, 0, 1]})
    requests = list(DecisionGroupedBatchSampler(rows, 4))
    assert requests[0].indices == (3, 1, 4, 2)
    assert requests[0].valid_count == 4
    assert requests[1].indices == (0, 0, 0, 0)
    assert requests[1].valid_count == 1


def test_train_validation_test_boundaries_are_disjoint() -> None:
    rows = pl.DataFrame(
        {
            "trade_date": [
                date(2024, 6, 28),
                date(2024, 7, 8),
                date(2025, 6, 30),
                date(2025, 7, 7),
            ],
            "sample_id": [0, 1, 2, 3],
        }
    )
    assert select_sample_split(rows, "train").get_column("sample_id").to_list() == [0]
    assert select_sample_split(rows, "validation").get_column(
        "sample_id"
    ).to_list() == [1, 2]
    assert select_sample_split(rows, "test").get_column("sample_id").to_list() == [3]
