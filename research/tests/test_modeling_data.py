from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from brazil_rv.modeling.contract import (
    ABSOLUTE_PATCH_COUNT,
    BASELINE_TCN_SETTINGS,
    EFFECTIVE_BATCH_SIZE,
    EQUITY_ABSOLUTE_START_PATCH,
    EQUITY_COUNT,
    GH200_RUNTIME,
    GLOBAL_CONTEXT_COUNT,
    HORIZON_COUNT,
    LOCAL_CONTEXT_COUNT,
    PATCH_INPUT_WIDTH,
    RuntimeSettings,
    context_family_slots,
    SLOW_FEATURE_COUNT,
    TABULAR_FEATURE_COUNT,
    TCNArchitecture,
    architecture_for_model,
)
from brazil_rv.modeling.data import (
    BatchRequest,
    DateStratifiedBatchSampler,
    DecisionGroupedBatchSampler,
    VectorizedFeatureDataset,
    _build_patch_batch,
    _build_peer_state,
    _validate_sample_index,
    build_tabular_batch,
    select_sample_split,
    tensorize_vectorized_batch,
)
from brazil_rv.modeling.engine import collect_validation_observations, objective_loss
from brazil_rv.modeling.model import build_neural_model


def _arrays() -> dict[str, np.ndarray]:
    equity_features = np.arange(EQUITY_COUNT * 405 * 26, dtype=np.float32).reshape(
        1, EQUITY_COUNT, 405, 26
    )
    context_features = np.ones((1, LOCAL_CONTEXT_COUNT, 465, 26), dtype=np.float32)
    global_features = np.ones((1, GLOBAL_CONTEXT_COUNT, 615, 26), dtype=np.float32)
    equity_features[..., 5] = 1
    context_features[..., 5] = 1
    global_features[..., 5] = 1
    targets = np.broadcast_to(
        np.linspace(-0.02, 0.02, EQUITY_COUNT, dtype=np.float32)[None, :, None, None],
        (1, EQUITY_COUNT, 55, HORIZON_COUNT),
    ).copy()
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
        "targets.npy": targets,
        "label_mask.npy": np.ones_like(targets, dtype=bool),
        "raw_returns.npy": targets / 10,
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


@pytest.mark.parametrize("family", ("wdo", "br_rates", "us_rates"))
def test_dataset_training_ablation_disables_family_before_model_input(
    family: str,
) -> None:
    rows = pl.DataFrame(
        {
            "sample_id": [0],
            "date_idx": [0],
            "decision_idx": [0],
            "equity_cutoff_index": [15],
            "context_cutoff_index": [75],
        }
    )
    architecture = architecture_for_model("tcn", BASELINE_TCN_SETTINGS)
    assert isinstance(architecture, TCNArchitecture)
    dataset = VectorizedFeatureDataset(
        Path("."),
        rows,
        "tcn",
        "enabled",
        architecture,
        "selected",
        family,
    )
    dataset._arrays = _arrays()
    batch = dataset[BatchRequest((0,), 1)]
    slots = tuple(EQUITY_COUNT + slot for slot in context_family_slots(family))
    for name in (
        "patches",
        "history_patch_mask",
        "instrument_mask",
        "slow_features",
    ):
        assert not batch[name][:, slots].any()
    assert batch["instrument_mask"][:, :EQUITY_COUNT].all()
    assert batch["peer_state"].shape[-1] == 6


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


def test_gh200_runtime_contract_and_derived_batch_counts() -> None:
    runtime = GH200_RUNTIME
    assert runtime.effective_batch_size == EFFECTIVE_BATCH_SIZE == 512
    assert runtime.loader_batch_size == 256
    assert runtime.microbatch_size == 256
    assert runtime.loader_batches_per_effective_batch == 2
    assert runtime.microbatches_per_effective_batch == 2
    assert runtime.evaluation_batch_size == 256
    assert (runtime.num_workers, runtime.prefetch_factor) == (8, 4)
    assert runtime.compile_backend == "inductor"
    assert runtime.compile_mode == "default"
    assert runtime.compile_fullgraph is True
    assert runtime.compile_dynamic is False


@pytest.mark.parametrize(
    ("settings", "message"),
    (
        ({"effective_batch_size": 0}, "positive"),
        (
            {"effective_batch_size": 6, "loader_batch_size": 4, "microbatch_size": 2},
            "divisible by loader",
        ),
        (
            {"effective_batch_size": 12, "loader_batch_size": 6, "microbatch_size": 5},
            "divisible by microbatch",
        ),
        (
            {"effective_batch_size": 4, "loader_batch_size": 1, "microbatch_size": 2},
            "at least",
        ),
        (
            {"effective_batch_size": 12, "loader_batch_size": 6, "microbatch_size": 4},
            "Loader batch size must be divisible",
        ),
    ),
)
def test_runtime_rejects_invalid_batch_sizes(
    settings: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RuntimeSettings(**settings)


def test_date_stratified_sampler_preserves_effective_sequence_and_reseeding() -> None:
    rows = _sample_index(EFFECTIVE_BATCH_SIZE)
    runtime = RuntimeSettings(num_workers=0)
    sampler = DateStratifiedBatchSampler(rows, runtime, seed=29)
    iterator = iter(sampler)
    requests = [
        next(iterator) for _ in range(runtime.loader_batches_per_effective_batch)
    ]
    assert [len(request.indices) for request in requests] == [256, 256]
    positions = [position for request in requests for position in request.indices]
    selected = rows[positions]
    assert selected.get_column("trade_date").n_unique() == EFFECTIVE_BATCH_SIZE
    decisions = selected.get_column("decision_idx").to_list()
    assert decisions == sorted(decisions)
    assert min(decisions) >= 0 and max(decisions) < 55

    legacy_runtime = RuntimeSettings(loader_batch_size=64, microbatch_size=64)
    legacy = DateStratifiedBatchSampler(rows, legacy_runtime, seed=29)
    legacy_iterator = iter(legacy)
    legacy_positions = [
        position
        for _ in range(legacy_runtime.loader_batches_per_effective_batch)
        for position in next(legacy_iterator).indices
    ]
    assert positions == legacy_positions
    repeated = DateStratifiedBatchSampler(rows, runtime, seed=29)
    repeated_iterator = iter(repeated)
    assert requests == [
        next(repeated_iterator)
        for _ in range(runtime.loader_batches_per_effective_batch)
    ]
    sampler.set_epoch(1)
    epoch_two = iter(sampler)
    assert requests != [
        next(epoch_two) for _ in range(runtime.loader_batches_per_effective_batch)
    ]


def test_padding_is_explicit_and_does_not_create_real_rows() -> None:
    rows = pl.DataFrame({"sample_id": [4, 2, 3, 1, 0], "decision_idx": [1, 0, 1, 0, 1]})
    requests = list(DecisionGroupedBatchSampler(rows, 4))
    assert requests[0].indices == (3, 1, 4, 2)
    assert requests[0].valid_count == 4
    assert requests[1].indices == (0, 0, 0, 0)
    assert requests[1].valid_count == 1


def test_padded_tcn_batch_reuses_safe_inputs_and_is_excluded_from_validation() -> None:
    rows = pl.DataFrame(
        {
            "sample_id": [2, 0, 1],
            "date_idx": [0, 0, 0],
            "decision_idx": [0, 2, 1],
            "equity_cutoff_index": [15, 25, 20],
            "context_cutoff_index": [75, 85, 80],
        }
    )
    architecture = architecture_for_model("tcn", BASELINE_TCN_SETTINGS)
    assert isinstance(architecture, TCNArchitecture)
    dataset = VectorizedFeatureDataset(
        Path("."), rows, "tcn", "enabled", architecture, "selected"
    )
    dataset._arrays = _arrays()
    request = next(iter(DecisionGroupedBatchSampler(rows, 4)))
    assert request.indices == (0, 2, 1, 1)
    assert request.valid_count == rows.height
    numpy_batch = dataset[request]
    padded = slice(request.valid_count, None)
    repeated = slice(request.valid_count - 1, request.valid_count)
    for name in (
        "patches",
        "history_patch_mask",
        "instrument_mask",
        "slow_features",
        "state_position",
        "peer_state",
    ):
        np.testing.assert_array_equal(
            numpy_batch[name][padded], numpy_batch[name][repeated]
        )
    assert (numpy_batch["state_position"][padded] > 0).all()
    assert not numpy_batch["sample_valid_mask"][padded].any()
    assert not numpy_batch["label_mask"][padded].any()
    assert not numpy_batch["targets"][padded].any()
    assert not numpy_batch["raw_returns"][padded].any()
    for name in ("sample_id", "date_idx", "decision_idx"):
        np.testing.assert_array_equal(numpy_batch[name][padded], [-1])

    batch = tensorize_vectorized_batch(numpy_batch)
    assert int((batch["state_position"] - 1).min()) >= 0
    model = build_neural_model("tcn", architecture, "selected").eval()
    observations, loss = collect_validation_observations(
        model, [batch], "soft_spearman", 0.50
    )
    assert observations.predictions.shape[0] == rows.height
    assert -1 not in observations.sample_id
    np.testing.assert_array_equal(observations.sample_id, [0, 1, 2])
    expected_loss = objective_loss(
        torch.from_numpy(observations.predictions),
        torch.from_numpy(observations.targets),
        torch.from_numpy(observations.label_mask),
        "soft_spearman",
        0.50,
    )
    assert np.isclose(loss, float(expected_loss))


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
