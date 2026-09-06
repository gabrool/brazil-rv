from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from brazil_rv.v2.data import V2DailyDataset, collate_v2_daily
from brazil_rv.v2.store import write_store


def _native_fast_store(tmp_path):
    dates = [date(2021, 7, 1) + timedelta(days=index) for index in range(21)]
    dates.append(date(2021, 8, 16))
    name_count = 3
    values = np.zeros((len(dates), 2, 5, 7), dtype=np.float32)
    valid = np.zeros_like(values, dtype=np.bool_)
    patch_mask = np.zeros(values.shape[:-1], dtype=np.bool_)
    patch_mask[-1, 0, :3] = True
    valid[-1, 0, :3] = True
    values[-1, 0, :3] = 2.0
    valid[-1, 0, 1, 4] = False
    values[-1, 0, 1, 4] = np.nan
    return write_store(
        tmp_path / "native_fast",
        dates=dates,
        isins=("BRTESTACNOR1", "BRTESTBCNOR2", "BRTESTCCNOR3"),
        arrays={
            "slow_values": np.zeros((len(dates), name_count, 1), dtype=np.float32),
            "slow_valid": np.ones((len(dates), name_count, 1), dtype=np.bool_),
            "active": np.ones((len(dates), name_count), dtype=np.bool_),
            "fast_patch_values": values,
            "fast_patch_valid": valid,
            "fast_patch_mask": patch_mask,
        },
        tables={
            "native_fast_security_mapping": pl.DataFrame(
                {
                    "fast_index": [0, 1],
                    "store_name_index": [2, 0],
                    "isin": ["BRTESTCCNOR3", "BRTESTACNOR1"],
                }
            )
        },
    )


def test_dataset_compacts_native_fast_slots_and_zeroes_invalid_payload(tmp_path) -> None:
    dataset = V2DailyDataset(
        _native_fast_store(tmp_path), [21], stage="finetune", lookback=20
    )
    sample = dataset[0]

    assert sample["fast_patch_values"].shape == (1, 5, 7)
    assert sample["fast_patch_valid"].shape == (1, 5, 7)
    assert sample["fast_patch_mask"].shape == (1, 5)
    assert sample["fast_name_index"].tolist() == [2]
    assert sample["fast_state_position"].tolist() == [3]
    assert sample["fast_present"].tolist() == [False, False, True]
    assert sample["fast_patch_values"][0, 1, 4] == 0.0
    assert np.isfinite(sample["fast_patch_values"]).all()


def test_pretraining_dataset_never_allocates_fast_name_rows(tmp_path) -> None:
    dataset = V2DailyDataset(
        _native_fast_store(tmp_path), [1], stage="pretrain", lookback=20
    )
    sample = dataset[0]
    batch = collate_v2_daily((sample, sample))

    assert sample["fast_patch_values"].shape == (0, 5, 7)
    assert sample["fast_name_index"].shape == (0,)
    assert batch["fast_patch_values"].shape == (2, 0, 5, 7)
    assert batch["fast_name_index"].shape == (2, 0)
