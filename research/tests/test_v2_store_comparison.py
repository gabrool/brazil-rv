from datetime import date

import numpy as np
import polars as pl
import pytest

from brazil_rv.v2.store import open_store_for_dates
from brazil_rv.v2.store_comparison import monthly_coverage
from v2_store_fixtures import write_fixture_store


def test_monthly_native_coverage_uses_identity_mapping_and_full_active_denominator(
    tmp_path,
):
    dates = [date(2024, 1, 2), date(2024, 2, 1)]
    isins = ["BRTESTACNOR1", "BRTESTBCNOR1", "BRTESTCCNOR1"]
    active = np.asarray([[True, True, False], [True, False, True]])
    valid = np.zeros((2, 1, 2, 7), dtype=np.bool_)
    valid[:, 0, 0, 0] = True
    valid[:, 0, :, 1] = True
    path = write_fixture_store(
        tmp_path / "store",
        dates=dates,
        isins=isins,
        arrays={
            "active": active,
            "fast_present": np.asarray([[False, True, False], [False, True, False]]),
            "fast_patch_values": np.zeros(valid.shape, dtype=np.float32),
            "fast_patch_valid": valid,
            "fast_patch_mask": np.ones((2, 1, 2), dtype=np.bool_),
        },
        tables={
            "native_fast_security_mapping": pl.DataFrame(
                {"fast_index": [0], "store_name_index": [1], "isin": [isins[1]]}
            )
        },
    )
    store, _ = open_store_for_dates(path, [0, 1], purpose="evaluation")
    try:
        january, february = monthly_coverage(
            store, np.asarray(dates, dtype="datetime64[D]")
        )
    finally:
        store.close()
    assert january["family"] == "native_fast"
    assert january["active_name_days"] == 2
    assert january["mapped_active_name_days"] == 1
    assert january["fast_present_active_name_days"] == 1
    assert january["features"]["native_fast_fixture_0"]["fraction"] == 0.5
    assert january["features"]["native_fast_fixture_1"]["fraction"] == 0.5
    assert january["mean_feature_coverage"] == pytest.approx(1 / 7)
    assert february["active_name_days"] == 2
    assert february["mapped_active_name_days"] == 0
    assert february["any_feature_valid_name_days"] == 0
    assert february["fast_present_active_name_days"] == 0
