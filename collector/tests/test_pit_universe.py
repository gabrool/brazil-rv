from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from pit_universe import (  # noqa: E402
    build_universe_tables,
    eligibility_result,
    interval_contains,
    is_recent_observation,
    median_with_absent_sessions,
    review_schedule,
    transition_action,
)


def passing_metrics() -> dict[str, object]:
    return {
        "accepted_identity": True,
        "age_sessions": 63,
        "presence_sessions": 57,
        "recent_observation": True,
        "last_close_brl": 1.0,
        "median_daily_turnover_brl": 2_000_000.0,
        "median_daily_trade_count": 200.0,
    }


def selected(**overrides: object) -> dict[str, bool | str]:
    values = passing_metrics()
    values.update(overrides)
    return eligibility_result(**values)


def test_exact_boundaries_pass() -> None:
    result = selected()
    assert result["equity_eligible"]
    assert result["selection_reason"] == "ELIGIBLE"


@pytest.mark.parametrize(
    ("override", "reason"),
    (
        ({"age_sessions": 62}, "INSUFFICIENT_HISTORY"),
        ({"presence_sessions": 56}, "INSUFFICIENT_PRESENCE"),
        ({"recent_observation": False}, "STALE"),
        ({"last_close_brl": 0.99}, "SUB_PRICE_FLOOR"),
        ({"median_daily_turnover_brl": 1_999_999.0}, "LOW_TURNOVER"),
        ({"median_daily_trade_count": 199.0}, "LOW_TRADE_COUNT"),
    ),
)
def test_each_boundary_fails_below_threshold(
    override: dict[str, object], reason: str
) -> None:
    result = selected(**override)
    assert not result["equity_eligible"]
    assert result["selection_reason"] == reason


def test_absent_sessions_are_zeros_in_medians() -> None:
    observed = [5_000_000.0] * 31
    assert median_with_absent_sessions(observed, 63) == 0.0
    assert median_with_absent_sessions([3_000_000.0] * 57, 63) == 3_000_000.0


def test_extreme_days_do_not_rescue_low_typical_liquidity() -> None:
    values = [1_000_000.0] * 60 + [1_000_000_000.0] * 3
    assert median_with_absent_sessions(values, 63) == 1_000_000.0


def test_recency_fifth_session_boundary() -> None:
    window = [date(2026, 1, 1) + timedelta(days=index) for index in range(63)]
    assert is_recent_observation(window[-5], window, 5)
    assert not is_recent_observation(window[-6], window, 5)


@pytest.mark.parametrize("qualifying_count", (20, 130, 180))
def test_no_quota_selects_every_qualifying_identity(
    qualifying_count: int,
) -> None:
    results = [selected() for _ in range(qualifying_count)]
    assert (
        sum(bool(result["equity_eligible"]) for result in results) == qualifying_count
    )


def test_no_fill_and_no_truncation() -> None:
    results = [selected(median_daily_turnover_brl=10_000_000.0) for _ in range(180)]
    results.append(selected(median_daily_turnover_brl=2_000_000.0))
    results.append(selected(median_daily_turnover_brl=1_999_999.0))
    assert sum(bool(result["equity_eligible"]) for result in results) == 181
    assert results[-2]["equity_eligible"]
    assert not results[-1]["equity_eligible"]


def test_unaccepted_identity_never_enters() -> None:
    result = selected(accepted_identity=False)
    assert not result["equity_eligible"]
    assert result["selection_reason"] == "UNACCEPTED_IDENTITY"


@pytest.mark.parametrize(
    ("was_member", "is_member", "action"),
    (
        (False, True, "ENTER"),
        (True, True, "STAY"),
        (True, False, "EXIT"),
        (False, False, "OUT"),
    ),
)
def test_transition_semantics(was_member: bool, is_member: bool, action: str) -> None:
    assert transition_action(was_member, is_member) == action


def test_membership_is_stateless() -> None:
    assert not selected(median_daily_trade_count=199.0)["equity_eligible"]
    assert selected()["equity_eligible"]
    assert transition_action(True, False) == "EXIT"
    assert transition_action(False, True) == "ENTER"


def business_days(start: date, count: int) -> list[date]:
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def test_implementation_lag_is_five_later_market_sessions() -> None:
    calendar = business_days(date(2021, 1, 4), 110)
    schedule = review_schedule(calendar)
    index = {trade_date: position for position, trade_date in enumerate(calendar)}
    assert schedule
    for review_date, effective_from in schedule:
        assert index[effective_from] - index[review_date] == 5


def test_intervals_use_exclusive_ends() -> None:
    start = date(2026, 1, 5)
    end = date(2026, 2, 5)
    assert interval_contains(start, start, end)
    assert interval_contains(end - timedelta(days=1), start, end)
    assert not interval_contains(end, start, end)
    assert interval_contains(end, start, None)


def synthetic_inputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    calendar = business_days(date(2021, 1, 4), 150)
    rows: list[dict[str, object]] = []
    identities = (
        ("ID:DIRECT", "DIRECT_ISIN", 3_000_000.0),
        ("ID:RECOVERED", "RECOVERED_RELABELED", 4_000_000.0),
        ("ID:UNACCEPTED", "UNRESOLVED", 20_000_000.0),
    )
    for trade_date in calendar:
        for security_id, _, turnover in identities:
            ticker = security_id.removeprefix("ID:")[:4]
            rows.append(
                {
                    "trade_date": trade_date,
                    "security_id": security_id,
                    "security_id_is_fallback": False,
                    "isin": security_id,
                    "ticker": ticker,
                    "issuer_short_name": ticker,
                    "security_spec": "ON",
                    "open_brl": 10.0,
                    "high_brl": 10.5,
                    "low_brl": 9.5,
                    "close_brl": 10.0,
                    "volume_brl": turnover,
                    "trades": 300,
                }
            )
    daily = pl.DataFrame(rows, infer_schema_length=None)
    observations = daily.select(
        "trade_date",
        "security_id",
        "security_id_is_fallback",
        "isin",
        "ticker",
        "issuer_short_name",
        "security_spec",
        "volume_brl",
    )
    assignments = pl.DataFrame(
        [
            {
                "security_id": security_id,
                "isin": security_id,
                "latest_ticker": security_id.removeprefix("ID:")[:4],
                "latest_issuer": security_id,
                "source_assignment_type": assignment_type,
            }
            for security_id, assignment_type, _ in identities[:2]
        ]
    )
    return daily, observations, assignments


def test_accepted_axis_identity_parity_and_output_reconciliation() -> None:
    daily, observations, assignments = synthetic_inputs()
    tables, _ = build_universe_tables(daily, observations, assignments)
    metrics = tables["universe_metrics_monthly"]
    membership = tables["universe_membership_monthly"]
    assert set(metrics["security_id"].unique()) == {"ID:DIRECT", "ID:RECOVERED"}
    assert set(membership["security_id"].unique()) == {
        "ID:DIRECT",
        "ID:RECOVERED",
    }
    assert membership.group_by("source_assignment_type").len().height == 2
    summary = tables["universe_summary"]
    assert summary["member_count"].to_list() == [2] * summary.height
    assert summary["equity_eligible_count"].to_list() == [2] * summary.height
    required = {
        "security_id": pl.String,
        "effective_from": pl.Date,
        "effective_to_exclusive": pl.Date,
        "is_member": pl.Boolean,
    }
    for column, dtype in required.items():
        assert membership.schema[column] == dtype


def test_row_order_and_future_mutation_invariance() -> None:
    daily, observations, assignments = synthetic_inputs()
    baseline, _ = build_universe_tables(daily, observations, assignments)
    reordered, _ = build_universe_tables(
        daily.reverse(), observations.reverse(), assignments.reverse()
    )
    for name in baseline:
        assert baseline[name].equals(reordered[name], null_equal=True)

    cutoff = baseline["universe_metrics_monthly"]["review_date"][0]
    changed_daily = daily.with_columns(
        pl.when(pl.col("trade_date") > cutoff)
        .then(pl.col("volume_brl") * 100.0)
        .otherwise(pl.col("volume_brl"))
        .alias("volume_brl")
    )
    changed_observations = observations.with_columns(
        pl.when(pl.col("trade_date") > cutoff)
        .then(pl.col("volume_brl") * 100.0)
        .otherwise(pl.col("volume_brl"))
        .alias("volume_brl")
    )
    changed, _ = build_universe_tables(changed_daily, changed_observations, assignments)
    baseline_early = baseline["universe_metrics_monthly"].filter(
        pl.col("review_date") <= cutoff
    )
    changed_early = changed["universe_metrics_monthly"].filter(
        pl.col("review_date") <= cutoff
    )
    assert baseline_early.equals(changed_early, null_equal=True)
