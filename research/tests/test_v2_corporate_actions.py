from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import polars as pl

from brazil_rv.v2 import corporate_actions as actions_module
from brazil_rv.v2.corporate_actions import (
    _extract_yfinance_actions,
    acquire_yfinance_actions,
    align_action_arrays,
    action_calendar_alignment_table,
    action_coverage_table,
    audit_m1_adjustment_status,
    causal_price_adjustment_factor,
    detect_cotahist_actions,
    detect_distribution_changes,
    normalize_cached_action_schema,
    normalize_yfinance_actions,
    unadjust_yfinance_cash_distributions,
)


def test_cotahist_split_detection_and_provider_alignment_are_independent() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    close = np.asarray([[100.0], [50.0], [51.0]])
    quantity = np.asarray([[1_000.0], [2_000.0], [2_100.0]])
    observed = np.ones_like(close, dtype=np.bool_)
    dismes = np.ones_like(close)
    detected = detect_cotahist_actions(close, quantity, dismes, observed)
    assert detected.split_event[:, 0].tolist() == [False, True, False]

    actions = normalize_yfinance_actions(
        pl.DataFrame(
            {
                "date": [dates[1]],
                "dividends": [0.5],
                "stock_splits": [2.0],
            }
        ),
        isin="BRTESTACNOR1",
        ticker="TEST3",
        fetched_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
    )
    split, cash, unresolved = align_action_arrays(
        actions, dates, ["BRTESTACNOR1"]
    )
    np.testing.assert_array_equal(split[:, 0], [1.0, 2.0, 1.0])
    np.testing.assert_array_equal(cash[:, 0], [0.0, 0.5, 0.0])
    assert not unresolved.any()


def test_forward_split_adjustment_never_rewrites_history() -> None:
    ratio = np.full((5, 1), np.nan)
    ratio[2, 0] = 0.5
    split = np.zeros_like(ratio, dtype=bool)
    split[2, 0] = True
    price = causal_price_adjustment_factor(ratio, split)
    np.testing.assert_array_equal(price[:, 0], [1.0, 1.0, 2.0, 2.0, 2.0])


def test_cotahist_classifies_cash_and_ambiguous_dismes_changes() -> None:
    close = np.asarray(
        [[100.0, 100.0], [100.0, 100.0], [95.0, 99.0], [95.0, 99.0]]
    )
    quantity = np.asarray(
        [[100.0, 100.0], [100.0, 100.0], [50.0, 100.0], [50.0, 100.0]]
    )
    dismes = np.asarray([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]])
    result = detect_cotahist_actions(
        close, quantity, dismes, np.ones_like(close, dtype=bool)
    )
    assert result.ambiguous_event[2, 0]
    assert result.cash_event[2, 1]
    assert not result.split_event[2].any()

    # A large price jump is a candidate even without a DISMES change. With no
    # offsetting quantity move it is cash-type, not a split adjustment.
    no_dismes = np.ones((4, 1))
    large_drop = np.asarray([[100.0], [100.0], [80.0], [80.0]])
    large_result = detect_cotahist_actions(
        large_drop,
        np.full_like(large_drop, 100.0),
        no_dismes,
        np.ones_like(large_drop, dtype=bool),
    )
    assert large_result.event_candidate[2, 0]
    assert large_result.cash_event[2, 0]

    moderate_drop = np.asarray([[100.0], [100.0], [94.0], [94.0]])
    moderate_result = detect_cotahist_actions(
        moderate_drop,
        np.asarray([[100.0], [100.0], [50.0], [50.0]]),
        no_dismes,
        np.ones_like(moderate_drop, dtype=bool),
    )
    assert moderate_result.event_candidate[2, 0]
    assert moderate_result.ambiguous_event[2, 0]


def test_legacy_canonical_cache_schema_is_upgraded_in_memory() -> None:
    fetched_at = datetime(2024, 2, 1, tzinfo=timezone.utc)
    legacy = pl.DataFrame(
        {
            "isin": ["BRTESTACNOR1"],
            "ex_date": [date(2024, 1, 2)],
            "action_type": ["dividend"],
            "split_factor": [1.0],
            "cash_distribution_brl": [0.5],
        }
    )
    upgraded = normalize_cached_action_schema(
        legacy,
        isin="BRTESTACNOR1",
        ticker="TEST3",
        fetched_at=fetched_at,
    )
    assert upgraded[0, "provider_cash_distribution_brl"] == 0.5
    assert upgraded[0, "cash_unit_adjustment_factor"] == 1.0
    assert upgraded[0, "known_date"] == date(2024, 1, 2)
    assert upgraded[0, "source_ticker"] == "TEST3"


def test_legacy_provider_cache_schema_is_normalized_in_memory() -> None:
    fetched_at = datetime(2024, 2, 1, tzinfo=timezone.utc)
    cached = pl.DataFrame(
        {
            "Date": [date(2024, 1, 15)],
            "Dividends": [0.5],
            "Stock Splits": [0.0],
        }
    )
    upgraded = normalize_cached_action_schema(
        cached,
        isin="BRTESTACNOR1",
        ticker="TEST3",
        fetched_at=fetched_at,
    )
    assert upgraded[0, "action_type"] == "dividend"
    assert upgraded[0, "cash_distribution_brl"] == 0.5
    assert upgraded[0, "source_ticker"] == "TEST3"


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
    assert extracted.height == 1
    assert extracted[0, "dividends"] == 0.0


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


def test_historical_segment_retries_under_current_isin_ticker(
    tmp_path, monkeypatch
) -> None:
    master = pl.DataFrame(
        {
            "isin": ["BRTESTACNOR1", "BRTESTACNOR1"],
            "ticker": ["OLD3", "NEW3"],
            "first_date": [date(2020, 1, 1), date(2024, 1, 1)],
            "last_date": [date(2020, 1, 31), date(2024, 1, 31)],
        }
    )

    def fake_download(tickers, *, start, end):
        del end
        ticker = tickers[0]
        if ticker == "OLD3":
            return {
                ticker: pl.DataFrame(
                    {
                        "date": [date(2024, 1, 10)],
                        "dividends": [0.0],
                        "stock_splits": [0.0],
                        "price_observed": [True],
                    }
                )
            }
        day = date(2020, 1, 10) if start.year == 2020 else date(2024, 1, 10)
        return {
            ticker: pl.DataFrame(
                {
                    "date": [day],
                    "dividends": [0.25 if day.year == 2020 else 0.0],
                    "stock_splits": [0.0],
                    "price_observed": [True],
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
    old = audit.filter(pl.col("ticker") == "OLD3").row(0, named=True)
    assert old["query_ticker"] == "NEW3"
    assert old["status"] == "downloaded_current_ticker"
    assert result.filter(pl.col("ex_date") == date(2020, 1, 10)).height == 1


def test_provider_failure_state_does_not_enter_cotahist_classification() -> None:
    observed = np.ones((5, 1), dtype=bool)
    dismes = np.asarray([[1.0], [1.0], [2.0], [2.0], [2.0]])
    result = detect_cotahist_actions(
        np.full((5, 1), 100.0),
        np.full((5, 1), 1_000.0),
        dismes,
        observed,
    )
    assert result.cash_event[:, 0].tolist() == [False, False, True, False, False]
    changed = detect_distribution_changes(dismes, observed)
    np.testing.assert_array_equal(changed, result.event_candidate)


def test_cash_units_use_only_strictly_later_split_factors() -> None:
    fetched_at = datetime(2024, 2, 1, tzinfo=timezone.utc)
    same_day = normalize_yfinance_actions(
        pl.DataFrame(
            {
                "date": [date(2020, 1, 2), date(2021, 1, 4)],
                "dividends": [1.0, 0.0],
                "stock_splits": [2.0, 5.0],
            }
        ),
        isin="BRTESTACNOR1",
        ticker="TEST3",
        fetched_at=fetched_at,
    )
    adjusted = unadjust_yfinance_cash_distributions(same_day)
    dividend = adjusted.filter(pl.col("action_type") == "dividend").row(
        0, named=True
    )
    assert dividend["provider_cash_distribution_brl"] == 1.0
    assert dividend["cash_unit_adjustment_factor"] == 5.0
    assert dividend["cash_distribution_brl"] == 5.0


def test_action_alignment_audit_counts_off_calendar_rows() -> None:
    actions = normalize_yfinance_actions(
        pl.DataFrame(
            {
                "date": [date(2024, 1, 6)],
                "dividends": [0.5],
                "stock_splits": [0.0],
            }
        ),
        isin="BRTESTACNOR1",
        ticker="TEST3",
        fetched_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
    )
    audit = action_calendar_alignment_table(
        actions, [date(2024, 1, 5), date(2024, 1, 8)], ["BRTESTACNOR1"]
    )
    assert audit.to_dicts() == [
        {
            "isin": "BRTESTACNOR1",
            "ex_date": date(2024, 1, 6),
            "action_type": "dividend",
            "reason": "off_calendar_ex_date",
        }
    ]


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
