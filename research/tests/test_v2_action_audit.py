from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import polars as pl

from brazil_rv.v2.action_audit import build_reclassification_audit
from brazil_rv.v2.corporate_actions import normalize_yfinance_actions


def test_reclassification_audit_applies_preregistered_fallback_rule() -> None:
    dates = np.arange("2024-01-02", "2024-01-09", dtype="datetime64[D]")
    close = np.ones((dates.size, 1), dtype=np.float64) * 100.0
    quantity = np.ones_like(close) * 100.0
    close[1:4] = 50.0
    quantity[1:4] = 200.0
    close[4:] = 500.0
    quantity[4:] = 20.0
    observed = np.ones_like(close, dtype=np.bool_)
    distribution = np.ones_like(close)
    distribution[1:] = 2.0
    actions = normalize_yfinance_actions(
        pl.DataFrame(
            {
                "Date": [date(2024, 1, 3), date(2024, 1, 6)],
                "Dividends": [0.0, 0.0],
                "Stock Splits": [2.0, 10.0],
            }
        ),
        isin="BRTESTACNOR1",
        ticker="TEST3",
        fetched_at=datetime(2024, 1, 10, tzinfo=timezone.utc),
    )
    acquisition = pl.DataFrame(
        {
            "isin": ["BRTESTACNOR1"],
            "first_date": [date(2024, 1, 2)],
            "last_date": [date(2024, 1, 8)],
            "status": ["downloaded"],
        }
    )
    target_validity = pl.DataFrame(
        {
            "year": [2024],
            "horizon_sessions": [1],
            "valid_target_name_days": [4],
            "observed_member_name_days": [5],
            "validity_ratio": [0.8],
        }
    )
    payload = build_reclassification_audit(
        dates=dates,
        isins=("BRTESTACNOR1",),
        raw_close=close,
        quantity=quantity,
        trades=np.ones_like(quantity) * 10.0,
        distribution_number=distribution,
        observed=observed,
        active=observed,
        old_cash_event=np.array(
            [[False], [False], [False], [True], [False], [False], [False]]
        ),
        old_distribution_change=np.zeros_like(observed),
        provider_actions=actions,
        acquisition_audit=acquisition,
        old_target_validity_by_year=target_validity,
    )

    assert payload["decision"]["legacy_undocumented_split_fallback"] is True
    assert payload["decision"]["canonical_price_ratio_adjustment_authorized"] is False
    assert payload["provider_split_comparison"]["recall_gain"] == 0.5
    assert payload["provider_split_comparison"]["precision_loss"] == 0.0
    assert (
        payload["defect_record"]["old_cash_events_without_dismes_change_fraction"]
        == 1.0
    )
    assert payload["defect_record"]["old_target_validity_by_horizon"] == [
        {
            "horizon_sessions": 1,
            "valid_target_name_days": 4,
            "observed_member_name_days": 5,
            "validity_ratio": 0.8,
        }
    ]
    breakdown = payload["corporate_action_reclassification_breakdown"]
    assert breakdown["schema"].endswith("BREAKDOWN_V2")
    assert len(breakdown["provider_splits"]) == 2
    assert breakdown["plus_or_minus_2_metrics"]["dismes_only"]["recall"] == 0.5
    assert (
        breakdown["plus_or_minus_2_metrics"]["dismes_plus_strict_fallback"]["recall"]
        == 1.0
    )
    assert not breakdown["decision"]["canonical_price_ratio_adjustment_authorized"]
    assert breakdown["decision"]["u2_development_inference_enabled"]
    assert breakdown["development_inference_quality"]["audit_only_not_a_gate"] is True
