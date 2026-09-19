from datetime import date

import numpy as np
import polars as pl
import pytest

from brazil_rv.v2.lending_archive import load_lending_borrow_panels
from brazil_rv.v2.lending_recovery import (
    prior_loan_references,
    recover_rates,
    recover_balances,
)
from test_v2_lending_archive import _balances, _rates, _write_archive


DATES = [date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10)]


def test_reference_is_prior_publication_and_never_inherits_another_identity():
    quotes = pl.DataFrame(
        {
            "trade_date": [DATES[0], DATES[2], DATES[2]],
            "isin": ["OLD", "OLD", "NEW"],
            "average_brl": [10.0, 12.0, 30.0],
        }
    )
    values, published = prior_loan_references(quotes, DATES, ["NEW", "OLD", "NONE"])
    np.testing.assert_equal(values[:, 1], [np.nan, 10.0, 10.0, 12.0])
    np.testing.assert_equal(values[:, 0], [np.nan, np.nan, np.nan, 30.0])
    assert np.isnan(values[:, 2]).all()
    assert published[2, 1] == np.datetime64(DATES[0])
    future = quotes.with_columns(
        pl.when(pl.col("trade_date") >= DATES[2])
        .then(999.0)
        .otherwise(pl.col("average_brl"))
        .alias("average_brl")
    )
    changed, _ = prior_loan_references(future, DATES, ["NEW", "OLD", "NONE"])
    np.testing.assert_equal(changed[:3], values[:3])
    assert changed[3, 1] == 999


def test_ambiguous_reference_is_not_silently_last_row_wins():
    quotes = pl.DataFrame(
        {
            "trade_date": [DATES[0], DATES[0]],
            "isin": ["A", "A"],
            "average_brl": [10.0, 11.0],
        }
    )
    with pytest.raises(ValueError, match="ambiguous"):
        prior_loan_references(quotes, DATES, ["A"])


def rate_rows():
    return (
        _rates(
            [DATES[0], DATES[1], DATES[-1]],
            [DATES[1], DATES[2], DATES[-1]],
            ["ISIN:BRA"] * 3,
            [0.0, 0.4, 0.7],
        )
        .with_columns(
            pl.Series("annual_donor_rate", [0.0, 0.3, 0.5]),
            pl.Series("registered_quantity", [20, 0, 10]),
        )
        .drop("available_date")
    )


def test_exact_rate_repair_zero_rate_and_end_boundary_preserve_old_other_keys():
    old = _rates(
        [DATES[0], DATES[0]],
        [DATES[1], DATES[1]],
        ["ISIN:BRA", "ISIN:BRB"],
        [0.02, 0.05],
    )
    new = recover_rates(old, rate_rows(), np.array(DATES, dtype="datetime64[D]"))
    assert new.height == 2
    a = new.filter(pl.col("security_id") == "ISIN:BRA").row(0, named=True)
    assert a["annual_taker_rate"] == a["annual_donor_rate"] == 0
    assert a["registered_quantity"] == 20
    assert a["available_date"] == DATES[1]
    assert (
        new.filter(pl.col("security_id") == "ISIN:BRB")
        .select(old.columns)
        .equals(old.tail(1))
    )


def test_future_registered_rates_cannot_change_earlier_panels(tmp_path):
    old = _rates([DATES[0]], [DATES[1]], ["ISIN:BRA"], [0.02])
    rows = rate_rows().with_columns(pl.lit(20).alias("registered_quantity"))
    balances = _balances([DATES[0]], [DATES[1]], ["ISIN:BRA"], [10])
    outputs = []
    for i, future in enumerate((0.4, 3.0)):
        changed = rows.with_columns(
            pl.when(pl.col("source_trade_date") == DATES[1])
            .then(future)
            .otherwise(pl.col("annual_taker_rate"))
            .alias("annual_taker_rate")
        )
        rates = recover_rates(old, changed, np.array(DATES, dtype="datetime64[D]"))
        root = tmp_path / str(i)
        sha = _write_archive(root, balances=balances, rates=rates)
        outputs.append(
            load_lending_borrow_panels(
                root,
                expected_manifest_sha256=sha,
                canonical_dates=DATES,
                canonical_isins=["BRA"],
            )
        )
    np.testing.assert_array_equal(
        outputs[0].annual_taker_rate[:2], outputs[1].annual_taker_rate[:2]
    )
    assert outputs[0].annual_taker_rate[2, 0] == 0.4
    assert outputs[1].annual_taker_rate[2, 0] == 3.0


def test_balance_uses_report_date_and_does_not_guess_unknown_identity():
    old = _balances([DATES[0]], [DATES[1]], ["ISIN:BRB"], [50])
    rows = pl.DataFrame(
        {
            "position_date": [DATES[0]] * 2,
            "report_date": [DATES[1]] * 2,
            "isin": ["BRA", None],
            "quote_isin": ["BRA", None],
            "quantity": [100, 999],
            "balance_brl": [200.0, 999.0],
        }
    )
    recovered = recover_balances(old, rows, np.array(DATES, dtype="datetime64[D]"))
    assert recovered.height == 2
    assert (
        recovered.filter(pl.col("security_id") == "ISIN:BRA")["available_date"][0]
        == DATES[2]
    )
    assert recovered.filter(pl.col("security_id") == "ISIN:BRB").equals(old)
    with pytest.raises(ValueError, match="identities conflict"):
        recover_balances(
            old,
            rows.with_columns(pl.lit("OTHER").alias("quote_isin")),
            np.array(DATES, dtype="datetime64[D]"),
        )


def test_conflicting_balance_cannot_overwrite_an_accepted_observation():
    old = _balances([DATES[0]], [DATES[1]], ["ISIN:BRA"], [50])
    rows = pl.DataFrame(
        {
            "position_date": [DATES[0]],
            "report_date": [DATES[0]],
            "isin": ["BRA"],
            "quote_isin": ["BRA"],
            "quantity": [100],
            "balance_brl": [200.0],
        }
    )
    with pytest.raises(ValueError, match="adjudication"):
        recover_balances(old, rows, np.array(DATES, dtype="datetime64[D]"))
