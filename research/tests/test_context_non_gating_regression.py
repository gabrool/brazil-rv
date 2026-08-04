from __future__ import annotations

import numpy as np

from brazil_rv.modeling.contract import (
    DYNAMIC_CHANNEL_COUNT,
    EQUITY_COUNT,
    GLOBAL_CONTEXT_COUNT,
    LOCAL_CONTEXT_COUNT,
    LOCAL_CONTEXT_SYMBOLS,
    SLOW_FEATURE_COUNT,
)
from brazil_rv.modeling.data import _build_patch_batch
from brazil_rv.preprocessing.build import _sample_date_is_eligible


def inputs() -> dict[str, np.ndarray]:
    return {
        "equity_features.npy": np.ones(
            (1, EQUITY_COUNT, 405, DYNAMIC_CHANNEL_COUNT), dtype=np.float32
        ),
        "equity_slow.npy": np.ones(
            (1, EQUITY_COUNT, SLOW_FEATURE_COUNT), dtype=np.float32
        ),
        "context_features.npy": np.ones(
            (1, LOCAL_CONTEXT_COUNT, 465, DYNAMIC_CHANNEL_COUNT),
            dtype=np.float32,
        ),
        "context_slow.npy": np.ones(
            (1, LOCAL_CONTEXT_COUNT, SLOW_FEATURE_COUNT), dtype=np.float32
        ),
        "context_data_ready.npy": np.ones((1, LOCAL_CONTEXT_COUNT), dtype=bool),
        "global_features.npy": np.ones(
            (1, GLOBAL_CONTEXT_COUNT, 615, DYNAMIC_CHANNEL_COUNT),
            dtype=np.float32,
        ),
        "global_slow.npy": np.ones(
            (1, GLOBAL_CONTEXT_COUNT, 55, SLOW_FEATURE_COUNT), dtype=np.float32
        ),
        "global_data_ready.npy": np.ones((1, GLOBAL_CONTEXT_COUNT, 55), dtype=bool),
    }


def batch(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return _build_patch_batch(
        arrays,
        date_idx=np.array([0]),
        equity_cutoffs=np.array([15]),
        decision_idx=np.array([0]),
        context_cutoffs=np.array([75]),
        active_equities=np.ones((1, EQUITY_COUNT), dtype=bool),
        global_context="enabled",
    )


def test_single_di_mask_changes_only_its_token_and_never_sample_eligibility() -> None:
    baseline_arrays = inputs()
    baseline = batch(baseline_arrays)
    changed_arrays = inputs()
    di_slot = LOCAL_CONTEXT_SYMBOLS.index("DI1F28")
    changed_arrays["context_data_ready.npy"][0, di_slot] = False
    changed = batch(changed_arrays)
    instrument = EQUITY_COUNT + di_slot

    assert _sample_date_is_eligible(EQUITY_COUNT)
    assert not changed["patches"][0, instrument].any()
    assert not changed["history_patch_mask"][0, instrument].any()
    assert not changed["slow_features"][0, instrument].any()
    assert not changed["instrument_mask"][0, instrument]
    for other in range(baseline["instrument_mask"].shape[1]):
        if other == instrument:
            continue
        np.testing.assert_array_equal(
            changed["patches"][0, other], baseline["patches"][0, other]
        )
        np.testing.assert_array_equal(
            changed["history_patch_mask"][0, other],
            baseline["history_patch_mask"][0, other],
        )
        np.testing.assert_array_equal(
            changed["slow_features"][0, other],
            baseline["slow_features"][0, other],
        )
        assert (
            changed["instrument_mask"][0, other]
            == baseline["instrument_mask"][0, other]
        )


def test_all_local_and_one_global_context_unavailable_remain_non_gating() -> None:
    arrays = inputs()
    arrays["context_data_ready.npy"][:] = False
    arrays["global_data_ready.npy"][0, 2, 0] = False
    changed = batch(arrays)
    local = slice(EQUITY_COUNT, EQUITY_COUNT + LOCAL_CONTEXT_COUNT)
    global_instrument = EQUITY_COUNT + LOCAL_CONTEXT_COUNT + 2

    assert _sample_date_is_eligible(EQUITY_COUNT)
    assert not changed["instrument_mask"][0, local].any()
    assert not changed["instrument_mask"][0, global_instrument]
    assert not changed["patches"][0, global_instrument].any()
    assert changed["instrument_mask"][0, :EQUITY_COUNT].all()
