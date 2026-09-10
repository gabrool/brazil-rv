import json
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from brazil_rv.v2.round5_derived import (
    beta_source_age,
    cdi_axis,
    identity_axes,
    joined_clock_fixture,
    write_family,
)
from brazil_rv.v2.round5_exposures import sector_relative_panel


def test_joined_market_clocks_enter_at_first_eligible_decision():
    assert len(joined_clock_fixture()["market_first_decision_mutations"]) == 2


def test_sector_identity_mutation_enters_on_receipt_without_prior_projection():
    dates = [date(2024, 1, 2) + timedelta(days=i) for i in range(4)]
    isins = ("A", "B", "C", "D")
    rows = [
        dict(
            date=day,
            isin=isin,
            cnpj=str(n) * 14,
            cvm_code=str(n),
            sector="original",
            identity_known_date=dates[0],
        )
        for day in dates
        for n, isin in enumerate(isins)
    ]
    values = np.broadcast_to(np.arange(4)[None, :, None], (4, 4, 3)).astype(float)
    valid, active = np.ones_like(values, bool), np.ones((4, 4), bool)
    sectors, issuers = identity_axes(pl.DataFrame(rows), dates, isins)
    before = sector_relative_panel(values, valid, sectors, issuers, active)
    for row in rows:
        if row["isin"] == "D" and row["date"] >= dates[2]:
            row.update(sector="changed", identity_known_date=dates[2])
    sectors, issuers = identity_axes(pl.DataFrame(rows), dates, isins)
    after = sector_relative_panel(values, valid, sectors, issuers, active)
    for left, right in zip(before, after):
        np.testing.assert_array_equal(left[:2], right[:2])
    assert before[0][2, 0, 0] != after[0][2, 0, 0]
    assert not after[1][2:, 3].any()
    rows[-1]["identity_known_date"] = dates[-1] + timedelta(days=1)
    with pytest.raises(ValueError, match="precedes"):
        identity_axes(pl.DataFrame(rows), dates, isins)


def test_beta_age_uses_last_pair_strictly_before_decision():
    wealth = np.array([[10], [11], [12], [np.nan], [14], [15.0]])
    seen = np.isfinite(wealth)
    valid = np.ones_like(seen)
    valid[:2] = False
    bova = np.arange(100, 106, dtype=float)
    age = beta_source_age(wealth, seen, bova, valid)
    assert age[:, 0].tolist() == [-1, -1, 1, 1, 2, 3]
    wealth[5] = 200
    np.testing.assert_array_equal(age, beta_source_age(wealth, seen, bova, valid))


def test_cdi_join_preserves_accepted_values_and_rejects_revision(tmp_path):
    days = [date(2016, 7, 18), date(2016, 7, 19)]
    early, accepted = tmp_path / "early.json", tmp_path / "accepted.parquet"
    early.write_text(json.dumps([{"data": "18/07/2016", "valor": "0.05"}]))
    pl.DataFrame(
        {"trade_date": days, "daily_cdi_rate": [0.0005, 0.0006]}
    ).write_parquet(accepted)
    values, audit = cdi_axis(early, accepted, days)
    assert values.tolist() == [0.0005, 0.0006]
    assert audit["overlap_rows"] == 1
    early.write_text(json.dumps([{"data": "18/07/2016", "valor": "0.06"}]))
    with pytest.raises(ValueError, match="conflicts"):
        cdi_axis(early, accepted, days)


def test_family_archive_keeps_missingness_and_physical_values(tmp_path):
    values = np.array([[[12.5], [99.0]], [[25.0], [0.0]]], np.float32)
    valid = np.array([[[True], [False]], [[True], [True]]])
    ages = np.array([[[0], [-1]], [[1], [0]]], np.float32)
    active = np.array([[True, True], [True, False]])
    result = write_family(
        tmp_path,
        "magnitudes",
        ("economic_beta_60",),
        values,
        valid,
        ages,
        [date(2024, 1, 2), date(2024, 1, 3)],
        ("A", "B"),
        active,
        {},
        {"commit": "fixture"},
        joined_clock_fixture(),
    )
    frame = pl.read_parquet(result["data"]["path"])
    assert frame["isin"].to_list() == ["A", "A"]
    assert frame["economic_beta_60"].to_list() == [12.5, 25.0]
    assert frame["economic_beta_60_age_sessions"].to_list() == [0, 1]
