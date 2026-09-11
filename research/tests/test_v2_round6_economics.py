from dataclasses import fields
from datetime import date

import numpy as np
import polars as pl
import pytest

from brazil_rv.v2.lending_archive import LendingBorrowPanels
from brazil_rv.v2.round6_economics import placeholder_v2


def test_hindsight_placeholder_only_changes_placeholder_costs_and_keeps_availability():
    shape = (3, 3)
    flags = np.zeros(shape, bool)
    placeholder = flags.copy()
    placeholder[:2] = True
    parent = LendingBorrowPanels(
        np.full(shape, 0.02),
        flags.copy(),
        placeholder,
        ~flags,
        ~flags,
        ~flags,
        "manifest",
        "balance",
        "rate",
        "source",
        (),
        (),
    )
    rates = pl.DataFrame(
        {
            "security_id": ["ISIN:A", "ISIN:B"],
            "source_trade_date": [date(2023, 2, 1)] * 2,
            "annual_taker_rate": [0.1, 0.3],
        }
    )
    days = np.array(["2020-01-02", "2023-01-02", "2024-01-02"], dtype="datetime64[D]")
    adv = np.full(shape, 100.0)
    result, report = placeholder_v2(parent, rates, days, ("A", "B", "C"), adv)
    assert result.annual_taker_rate[0].tolist() == pytest.approx(
        [(0.1 + 20 * 0.2) / 21, (0.3 + 20 * 0.2) / 21, 0.2]
    )
    assert np.array_equal(result.annual_taker_rate[2], parent.annual_taker_rate[2])
    assert np.array_equal(parent.annual_taker_rate, np.full(shape, 0.02))
    assert report["promotion_weight"] == 0
    for f in fields(parent):
        if f.name != "annual_taker_rate":
            assert getattr(result, f.name) is getattr(parent, f.name)
    # Unrequested later observations cannot influence the 2023-2024 calibration.
    later = pl.DataFrame(
        {
            "security_id": ["ISIN:A"],
            "source_trade_date": [date(2025, 1, 2)],
            "annual_taker_rate": [100.0],
        }
    )
    changed, _ = placeholder_v2(
        parent, pl.concat([rates, later]), days, ("A", "B", "C"), adv
    )
    np.testing.assert_array_equal(result.annual_taker_rate, changed.annual_taker_rate)


def test_placeholder_rejects_conflicting_rate_revisions():
    rates = pl.DataFrame(
        {
            "security_id": ["ISIN:A"] * 2,
            "source_trade_date": [date(2023, 2, 1)] * 2,
            "annual_taker_rate": [0.1, 0.2],
        }
    )
    with pytest.raises(ValueError, match="conflicting"):
        placeholder_v2(
            None, rates, np.array([], dtype="datetime64[D]"), (), np.empty((0, 0))
        )
