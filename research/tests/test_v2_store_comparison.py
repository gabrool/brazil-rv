from datetime import date

import numpy as np
import polars as pl
import pytest

from brazil_rv.v2.store import open_store_for_dates
from brazil_rv.v2.store_comparison import compare, monthly_coverage
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


def test_comparison_covers_embargo_history_and_censors_post_development_targets(
    tmp_path,
):
    dates = [date(2021, 7, 30), date(2021, 8, 2), date(2021, 8, 16), date(2024, 12, 30)]

    def write(label, *, future=False, changed=False):
        axis = [*dates, date(2025, 1, 2)] if future else dates
        count = len(axis)
        probe = np.arange(count, dtype=np.float32)[:, None]
        if changed:
            probe[1] += 1  # An embargo history row must still be compared.
        targets = np.ones((count, 1, 5), dtype=np.float32)
        targets[3] = 999 if future else 555  # All endpoints fall outside the grant.
        return write_fixture_store(
            tmp_path / label,
            dates=axis,
            isins=["BRTESTACNOR1"],
            arrays={
                "active": np.ones((count, 1), dtype=np.bool_),
                "probe": probe,
                "target_shareholder_simple_return": targets,
                "target_shareholder_midrank": np.full(
                    targets.shape, 0.5, dtype=np.float32
                ),
                "target_shareholder_valid": np.ones(targets.shape, dtype=np.bool_),
            },
        )

    previous = write("previous", future=True)
    current = write("current")
    result = compare(previous=previous, current=current, output=tmp_path / "exact")
    assert result["status"] == "passed"
    assert not result["previous_access"]["official_validation_accessed"]
    assert not result["current_access"]["official_validation_accessed"]
    changed = write("changed", changed=True)
    difference = compare(previous=previous, current=changed, output=tmp_path / "diff")
    assert difference["unexpected_differences"] == ["probe"]
    probe = next(row for row in difference["arrays"] if row["array"] == "probe")
    assert probe["different_cells"] == 1
    assert probe["examples"][0]["date"] == "2021-08-02"
