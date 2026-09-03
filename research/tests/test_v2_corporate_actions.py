from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import polars as pl

from brazil_rv.v2 import corporate_actions as actions_module
from brazil_rv.v2.corporate_actions import (
    _extract_yfinance_actions,
    acquire_yfinance_actions,
    action_coverage_table,
    audit_m1_adjustment_status,
    cash_reinvestment_review_table,
    cash_reinvestment_unavailable_mask,
    causal_adjustment_factors,
    detect_distribution_changes,
    normalize_yfinance_actions,
    provider_failure_mask,
)


def test_forward_adjustment_never_rewrites_history_and_retains_recorded_actions() -> None:
    close = np.full((5, 1), 10.0)
    split = np.ones_like(close)
    split[2, 0] = 2.0
    cash = np.zeros_like(close)
    cash[3, 0] = 1.0
    price, total_return = causal_adjustment_factors(close, split, cash)
    np.testing.assert_array_equal(price[:, 0], [1.0, 1.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(total_return[:, 0], [1.0, 1.0, 2.0, 2.2, 2.2])

    unresolved = np.zeros_like(close, dtype=bool)
    unresolved[2, 0] = True
    flagged, _ = causal_adjustment_factors(close, split, cash, unresolved)
    np.testing.assert_array_equal(flagged[:, 0], price[:, 0])


def test_missing_cash_reinvestment_close_requires_explicit_unresolved_flag() -> None:
    close = np.asarray([[10.0], [np.nan], [12.0]])
    split = np.ones_like(close)
    split[1, 0] = 2.0
    cash = np.asarray([[0.0], [1.0], [0.0]])
    unavailable = cash_reinvestment_unavailable_mask(close, cash)
    assert unavailable[:, 0].tolist() == [False, True, False]
    with np.testing.assert_raises_regex(ValueError, "positive ex-date close"):
        causal_adjustment_factors(close, split, cash)

    price, total_return = causal_adjustment_factors(
        close, split, cash, unavailable
    )
    np.testing.assert_array_equal(price[:, 0], [1.0, 2.0, 2.0])
    np.testing.assert_array_equal(total_return[:, 0], [1.0, 2.0, 2.0])
    review = cash_reinvestment_review_table(
        [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
        ["BRTESTACNOR1"],
        cash,
        close,
    )
    assert review.to_dicts() == [
        {
            "trade_date": date(2024, 1, 3),
            "isin": "BRTESTACNOR1",
            "cash_distribution_brl": 1.0,
            "raw_close_brl": None,
            "observed": False,
            "unresolved": True,
            "status": "unresolved_missing_ex_date_close",
        }
    ]


def test_normalize_yfinance_emits_split_and_dividend_rows() -> None:
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 2)],
            "dividends": [0.5],
            "stock_splits": [2.0],
        }
    )
    result = normalize_yfinance_actions(
        frame,
        isin="BRTESTACNOR1",
        ticker="TEST3",
        fetched_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
    )
    assert set(result.get_column("action_type")) == {"split", "dividend"}


def test_yfinance_nan_cells_are_not_actions() -> None:
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 3)],
            "dividends": [float("nan"), 0.5],
            "stock_splits": [float("nan"), float("nan")],
        }
    )
    result = normalize_yfinance_actions(
        frame,
        isin="BRTESTACNOR1",
        ticker="TEST3",
        fetched_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
    )
    assert result.height == 1
    assert result[0, "action_type"] == "dividend"
    assert result[0, "cash_distribution_brl"] == 0.5


def test_yfinance_all_nan_symbol_frame_is_provider_failure() -> None:
    columns = pd.MultiIndex.from_product(
        [["FAIL3.SA"], ["Close", "Dividends", "Stock Splits"]]
    )
    missing = pd.DataFrame(
        [[np.nan, np.nan, np.nan]],
        index=pd.DatetimeIndex(["2024-01-02"]),
        columns=columns,
    )
    assert _extract_yfinance_actions(missing, "FAIL3.SA") is None

    present = missing.copy()
    present.loc[:, ("FAIL3.SA", "Close")] = 10.0
    extracted = _extract_yfinance_actions(present, "FAIL3.SA")
    assert extracted is not None
    assert extracted.is_empty()


def test_batched_acquisition_records_symbol_failure_without_aborting(
    tmp_path, monkeypatch
) -> None:
    master = pl.DataFrame(
        {
            "isin": ["BRTESTACNOR1", "BRTESTACNPR0"],
            "ticker": ["GOOD3", "BAD4"],
            "first_date": [date(2024, 1, 1), date(2024, 1, 1)],
            "last_date": [date(2024, 1, 31), date(2024, 1, 31)],
        }
    )

    def fake_download(tickers, *, start, end):
        del start, end
        if tickers == ["BAD4"]:
            raise RuntimeError("provider failure")
        return {
            "GOOD3": pl.DataFrame(
                {
                    "date": [date(2024, 1, 10)],
                    "dividends": [0.25],
                    "stock_splits": [0.0],
                }
            )
        }

    monkeypatch.setattr(actions_module, "_download_yfinance_batch", fake_download)
    result, audit = acquire_yfinance_actions(
        master,
        tmp_path,
        fetched_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
        batch_size=1,
    )
    assert result.height == 1
    assert result[0, "isin"] == "BRTESTACNOR1"
    assert set(audit.get_column("status")) == {"downloaded", "failed"}
    assert audit.filter(pl.col("status") == "failed")[0, "cache_path"] is None


def test_provider_failures_and_distribution_changes_are_unresolved() -> None:
    dates = np.asarray(["2024-01-02", "2024-01-03", "2024-01-04"], dtype="datetime64[D]")
    observed = np.ones((3, 1), dtype=bool)
    audit = pl.DataFrame(
        {
            "isin": ["BRTESTACNOR1"],
            "first_date": [date(2024, 1, 2)],
            "last_date": [date(2024, 1, 4)],
            "status": ["failed"],
        }
    )
    failed = provider_failure_mask(audit, dates, ["BRTESTACNOR1"], observed)
    assert failed[:, 0].all()
    changed = detect_distribution_changes(
        np.asarray([[1.0], [1.0], [2.0]]), observed
    )
    assert changed[:, 0].tolist() == [False, False, True]


def test_action_coverage_distinguishes_true_zero_from_provider_failure() -> None:
    actions = normalize_yfinance_actions(
        pl.DataFrame(
            {
                "date": [date(2024, 6, 3)],
                "dividends": [0.25],
                "stock_splits": [0.0],
            }
        ),
        isin="BRTESTACNOR1",
        ticker="TEST3",
        fetched_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
    )
    audit = pl.DataFrame(
        {
            "isin": ["BRTESTACNOR1", "BRTESTACNPR0"],
            "first_date": [date(2024, 1, 1), date(2024, 1, 1)],
            "last_date": [date(2024, 12, 31), date(2024, 12, 31)],
            "status": ["downloaded", "zero_actions"],
            "action_rows": [1, 0],
        }
    )
    table = action_coverage_table(
        actions,
        [date(2024, 1, 2)],
        ["BRTESTACNOR1", "BRTESTACNPR0"],
        audit,
    )
    assert table[0, "acquisition_status"] == "covered_actions"
    assert table[1, "acquisition_status"] == "covered_zero_actions"
    failed = audit.with_columns(
        pl.when(pl.col("isin") == "BRTESTACNPR0")
        .then(pl.lit("failed"))
        .otherwise(pl.col("status"))
        .alias("status")
    )
    table = action_coverage_table(
        actions,
        [date(2024, 1, 2)],
        ["BRTESTACNOR1", "BRTESTACNPR0"],
        failed,
    )
    assert table[1, "acquisition_status"] == "provider_failure"


def test_m1_adjustment_audit_uses_pre_post_event_ratios() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    raw = np.asarray([[100.0], [50.0]])
    adjusted = np.asarray([[100.0], [100.0]])
    split = np.asarray([[1.0], [2.0]])
    cash = np.zeros_like(raw)
    raw_m1 = raw.copy()
    report = audit_m1_adjustment_status(
        dates,
        ["BRTESTACNOR1"],
        raw_m1,
        raw,
        adjusted,
        split,
        cash,
    )
    assert report[0, "prior_trade_date"] == dates[0]
    assert report[0, "m1_pre_post_ratio"] == 0.5
    assert report[0, "status"] == "raw_unadjusted"

    adjusted_m1 = adjusted.copy()
    report = audit_m1_adjustment_status(
        dates,
        ["BRTESTACNOR1"],
        adjusted_m1,
        raw,
        adjusted,
        split,
        cash,
    )
    assert report[0, "status"] == "price_adjusted"
