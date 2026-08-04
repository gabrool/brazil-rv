from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import polars as pl

ACCEPTED_ASSIGNMENT_COUNT = 158
DIRECT_ASSIGNMENT_COUNT = 143
RECOVERED_ASSIGNMENT_COUNT = 15
MIN_RESEARCH_CROSS_SECTION = 30

ELIGIBILITY_COLUMNS = (
    "eligible_history",
    "eligible_presence",
    "eligible_recency",
    "eligible_price",
    "eligible_turnover",
    "eligible_trade_count",
)
SELECTION_REASON_PRIORITY = (
    "UNACCEPTED_IDENTITY",
    "INSUFFICIENT_HISTORY",
    "INSUFFICIENT_PRESENCE",
    "STALE",
    "SUB_PRICE_FLOOR",
    "LOW_TURNOVER",
    "LOW_TRADE_COUNT",
    "ELIGIBLE",
)


@dataclass(frozen=True)
class UniverseConfig:
    start_date: date = date(2021, 7, 19)
    end_date: date = date(2026, 7, 17)
    lookback_sessions: int = 63
    implementation_lag_sessions: int = 5
    minimum_presence_sessions: int = 57
    recency_sessions: int = 5
    minimum_price_brl: float = 1.0
    minimum_median_daily_turnover_brl: float = 2_000_000.0
    minimum_median_daily_trade_count: int = 200

    def manifest_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["start_date"] = self.start_date.isoformat()
        values["end_date"] = self.end_date.isoformat()
        return values


CANONICAL_CONFIG = UniverseConfig()


def month_ends(calendar: list[date]) -> list[date]:
    result: list[date] = []
    for trade_date in calendar:
        key = (trade_date.year, trade_date.month)
        if result and (result[-1].year, result[-1].month) == key:
            result[-1] = trade_date
        else:
            result.append(trade_date)
    return result


def review_schedule(
    calendar: list[date], config: UniverseConfig = CANONICAL_CONFIG
) -> list[tuple[date, date]]:
    calendar_index = {trade_date: index for index, trade_date in enumerate(calendar)}
    schedule: list[tuple[date, date]] = []
    for review_date in month_ends(calendar):
        review_index = calendar_index[review_date]
        if review_index + 1 < config.lookback_sessions:
            continue
        effective_index = review_index + config.implementation_lag_sessions
        if effective_index >= len(calendar):
            continue
        effective_from = calendar[effective_index]
        if effective_from <= config.end_date:
            schedule.append((review_date, effective_from))
    if not schedule:
        raise ValueError("No review schedule could be created")
    return schedule


def median_with_absent_sessions(
    observed_values: list[float], window_sessions: int
) -> float:
    if len(observed_values) > window_sessions:
        raise ValueError("Observed values exceed the review window")
    return float(
        statistics.median(
            [0.0] * (window_sessions - len(observed_values)) + observed_values
        )
    )


def is_recent_observation(
    last_observed_date: date | None,
    window_dates: list[date],
    recency_sessions: int,
) -> bool:
    return bool(
        last_observed_date is not None
        and last_observed_date in window_dates[-recency_sessions:]
    )


def eligibility_result(
    *,
    accepted_identity: bool,
    age_sessions: int,
    presence_sessions: int,
    recent_observation: bool,
    last_close_brl: float | None,
    median_daily_turnover_brl: float,
    median_daily_trade_count: float,
    config: UniverseConfig = CANONICAL_CONFIG,
) -> dict[str, bool | str]:
    flags = {
        "eligible_history": age_sessions >= config.lookback_sessions,
        "eligible_presence": presence_sessions >= config.minimum_presence_sessions,
        "eligible_recency": recent_observation,
        "eligible_price": bool(
            last_close_brl is not None
            and math.isfinite(last_close_brl)
            and last_close_brl >= config.minimum_price_brl
        ),
        "eligible_turnover": bool(
            math.isfinite(median_daily_turnover_brl)
            and median_daily_turnover_brl >= config.minimum_median_daily_turnover_brl
        ),
        "eligible_trade_count": bool(
            math.isfinite(median_daily_trade_count)
            and median_daily_trade_count >= config.minimum_median_daily_trade_count
        ),
    }
    equity_eligible = accepted_identity and all(flags.values())
    reason_by_flag = (
        ("eligible_history", "INSUFFICIENT_HISTORY"),
        ("eligible_presence", "INSUFFICIENT_PRESENCE"),
        ("eligible_recency", "STALE"),
        ("eligible_price", "SUB_PRICE_FLOOR"),
        ("eligible_turnover", "LOW_TURNOVER"),
        ("eligible_trade_count", "LOW_TRADE_COUNT"),
    )
    if not accepted_identity:
        reason = "UNACCEPTED_IDENTITY"
    else:
        reason = next(
            (reason for flag, reason in reason_by_flag if not flags[flag]),
            "ELIGIBLE",
        )
    return {
        **flags,
        "equity_eligible": equity_eligible,
        "selection_reason": reason,
    }


def transition_action(was_member: bool, is_member: bool) -> str:
    if is_member:
        return "STAY" if was_member else "ENTER"
    return "EXIT" if was_member else "OUT"


def interval_contains(
    trade_date: date, effective_from: date, effective_to_exclusive: date | None
) -> bool:
    return effective_from <= trade_date and (
        effective_to_exclusive is None or trade_date < effective_to_exclusive
    )


def valid_observation_expr() -> pl.Expr:
    positive_ohlc = pl.all_horizontal(
        [
            pl.col(column).is_finite() & (pl.col(column) > 0)
            for column in ("open_brl", "high_brl", "low_brl", "close_brl")
        ]
    )
    return (
        positive_ohlc
        & pl.col("volume_brl").is_finite()
        & (pl.col("volume_brl") >= 0)
        & (pl.col("trades") >= 0)
    )


def validate_daily(daily: pl.DataFrame) -> pl.DataFrame:
    required = {
        "trade_date",
        "security_id",
        "isin",
        "ticker",
        "issuer_short_name",
        "security_spec",
        "open_brl",
        "high_brl",
        "low_brl",
        "close_brl",
        "volume_brl",
        "trades",
    }
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"COTAHIST daily input is missing columns: {missing}")
    duplicate_count = (
        daily.height - daily.unique(subset=["trade_date", "security_id"]).height
    )
    if duplicate_count:
        raise ValueError(
            f"COTAHIST daily input has duplicate security/date rows: {duplicate_count}"
        )
    return daily.sort(["trade_date", "security_id"])


def validate_accepted_assignments(assignments: pl.DataFrame) -> pl.DataFrame:
    required = {
        "security_id",
        "isin",
        "latest_ticker",
        "source_file",
        "source_assignment_type",
        "manual_decision",
        "normalization_rule",
    }
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise ValueError(f"Accepted assignments are missing columns: {missing}")
    assignments = assignments.sort("security_id")
    if (
        assignments.height != ACCEPTED_ASSIGNMENT_COUNT
        or assignments["security_id"].n_unique() != ACCEPTED_ASSIGNMENT_COUNT
    ):
        raise ValueError(
            "Accepted assignments must contain exactly 158 unique security IDs"
        )
    type_counts = {
        assignment_type: count
        for assignment_type, count in assignments.group_by("source_assignment_type")
        .len()
        .iter_rows()
    }
    expected = {
        "DIRECT_ISIN": DIRECT_ASSIGNMENT_COUNT,
        "RECOVERED_RELABELED": RECOVERED_ASSIGNMENT_COUNT,
    }
    if type_counts != expected:
        raise ValueError(
            f"Accepted assignment composition is {type_counts}, expected {expected}"
        )
    if not assignments["manual_decision"].eq("ACCEPTED").all():
        raise ValueError("Every accepted assignment must have manual_decision=ACCEPTED")
    normalization = "FILTER_TO_COTAHIST_SECURITY_DATES"
    if not assignments["normalization_rule"].eq(normalization).all():
        raise ValueError(
            f"Every accepted assignment must use normalization_rule={normalization}"
        )
    return assignments


def ticker_segments(observations: pl.DataFrame) -> pl.DataFrame:
    chosen = (
        observations.sort(["security_id", "trade_date", "volume_brl"])
        .group_by(["security_id", "trade_date"], maintain_order=True)
        .agg(
            pl.col("ticker").last(),
            pl.col("isin").last(),
            pl.col("issuer_short_name").last(),
            pl.col("security_spec").last(),
            pl.col("security_id_is_fallback").last(),
        )
        .sort(["security_id", "trade_date"])
    )
    rows: list[dict[str, object]] = []
    for group in chosen.partition_by("security_id", maintain_order=True):
        data = group.to_dicts()
        if not data:
            continue
        start = data[0]["trade_date"]
        prior = data[0]
        for current in data[1:]:
            if current["ticker"] != prior["ticker"]:
                rows.append(
                    {
                        "security_id": prior["security_id"],
                        "ticker": prior["ticker"],
                        "valid_from": start,
                        "valid_to": prior["trade_date"],
                        "isin": prior["isin"],
                        "issuer_short_name": prior["issuer_short_name"],
                        "security_spec": prior["security_spec"],
                        "security_id_is_fallback": prior["security_id_is_fallback"],
                    }
                )
                start = current["trade_date"]
            prior = current
        rows.append(
            {
                "security_id": prior["security_id"],
                "ticker": prior["ticker"],
                "valid_from": start,
                "valid_to": prior["trade_date"],
                "isin": prior["isin"],
                "issuer_short_name": prior["issuer_short_name"],
                "security_spec": prior["security_spec"],
                "security_id_is_fallback": prior["security_id_is_fallback"],
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        ["security_id", "valid_from"]
    )


def security_master(daily: pl.DataFrame, segments: pl.DataFrame) -> pl.DataFrame:
    latest = (
        daily.sort(["security_id", "trade_date", "volume_brl"])
        .group_by("security_id", maintain_order=True)
        .agg(
            pl.col("trade_date").min().alias("first_observed_date"),
            pl.col("trade_date").max().alias("last_observed_date"),
            pl.col("ticker").last().alias("latest_ticker"),
            pl.col("issuer_short_name").last().alias("latest_issuer_short_name"),
            pl.col("security_spec").last().alias("latest_security_spec"),
            pl.col("isin").last().alias("isin"),
            pl.col("security_id_is_fallback").max().alias("security_id_is_fallback"),
            pl.col("trade_date").n_unique().alias("observed_sessions"),
            pl.col("volume_brl").sum().alias("total_volume_brl"),
        )
    )
    ticker_info = segments.group_by("security_id").agg(
        pl.col("ticker").n_unique().alias("distinct_tickers"),
        pl.col("ticker").unique().sort().alias("tickers"),
    )
    return latest.join(ticker_info, on="security_id", how="left").sort("security_id")


def _review_records(
    *,
    valid_daily: pl.DataFrame,
    assignments: pl.DataFrame,
    calendar: list[date],
    first_seen: dict[str, date],
    review_date: date,
    effective_from: date,
    effective_to_exclusive: date | None,
    prior_members: set[str],
    config: UniverseConfig,
) -> list[dict[str, object]]:
    calendar_index = {trade_date: index for index, trade_date in enumerate(calendar)}
    review_index = calendar_index[review_date]
    window_dates = calendar[
        review_index - config.lookback_sessions + 1 : review_index + 1
    ]
    window = valid_daily.filter(pl.col("trade_date").is_in(window_dates))
    market_turnover_brl = math.fsum(window["volume_brl"].to_list())
    accepted_ids = assignments["security_id"].to_list()
    accepted_window = window.filter(pl.col("security_id").is_in(accepted_ids))
    observations_by_id: dict[str, list[dict[str, Any]]] = {}
    for group in accepted_window.partition_by("security_id", maintain_order=True):
        rows = group.sort("trade_date").to_dicts()
        observations_by_id[str(rows[0]["security_id"])] = rows
    assignment_by_id = {str(row["security_id"]): row for row in assignments.to_dicts()}

    records: list[dict[str, object]] = []
    for security_id in accepted_ids:
        assignment = assignment_by_id[security_id]
        observations = observations_by_id.get(security_id, [])
        presence_sessions = len({row["trade_date"] for row in observations})
        turnover_values = [float(row["volume_brl"]) for row in observations]
        trade_values = [float(row["trades"]) for row in observations]
        median_turnover = median_with_absent_sessions(
            turnover_values, config.lookback_sessions
        )
        median_trades = median_with_absent_sessions(
            trade_values, config.lookback_sessions
        )
        last = observations[-1] if observations else None
        last_observed_date = last["trade_date"] if last else None
        last_close_brl = float(last["close_brl"]) if last else None
        first_observed = first_seen.get(security_id)
        age_sessions = (
            review_index - calendar_index[first_observed] + 1
            if first_observed in calendar_index
            and calendar_index[first_observed] <= review_index
            else 0
        )
        eligibility = eligibility_result(
            accepted_identity=True,
            age_sessions=age_sessions,
            presence_sessions=presence_sessions,
            recent_observation=is_recent_observation(
                last_observed_date, window_dates, config.recency_sessions
            ),
            last_close_brl=last_close_brl,
            median_daily_turnover_brl=median_turnover,
            median_daily_trade_count=median_trades,
            config=config,
        )
        is_member = bool(eligibility["equity_eligible"])
        was_member = security_id in prior_members
        records.append(
            {
                "review_date": review_date,
                "effective_from": effective_from,
                "effective_to_exclusive": effective_to_exclusive,
                "window_start": window_dates[0],
                "window_end": review_date,
                "security_id": security_id,
                "isin": str((last or {}).get("isin") or assignment["isin"] or ""),
                "ticker_asof": str(
                    (last or {}).get("ticker") or assignment.get("latest_ticker") or ""
                ),
                "issuer_short_name_asof": str(
                    (last or {}).get("issuer_short_name")
                    or assignment.get("latest_issuer")
                    or ""
                ),
                "security_spec_asof": str((last or {}).get("security_spec") or ""),
                "source_assignment_type": assignment["source_assignment_type"],
                "accepted_identity": True,
                "age_sessions": age_sessions,
                "presence_sessions": presence_sessions,
                "presence_ratio": presence_sessions / config.lookback_sessions,
                "turnover_brl": float(sum(turnover_values)),
                "market_turnover_brl": market_turnover_brl,
                "market_share": (
                    float(sum(turnover_values)) / market_turnover_brl
                    if market_turnover_brl > 0
                    else 0.0
                ),
                "last_observed_date": last_observed_date,
                "last_close_brl": last_close_brl,
                "median_daily_turnover_brl": median_turnover,
                "median_daily_trade_count": median_trades,
                **eligibility,
                "was_member": was_member,
                "is_member": is_member,
                "action": transition_action(was_member, is_member),
            }
        )
    return records


def _summary(metrics: pl.DataFrame) -> pl.DataFrame:
    member = pl.col("is_member")
    return (
        metrics.group_by(["review_date", "effective_from"], maintain_order=True)
        .agg(
            pl.col("effective_to_exclusive").first(),
            pl.len().alias("accepted_identity_count"),
            (pl.col("presence_sessions") > 0)
            .sum()
            .alias("contemporaneously_observed_count"),
            *[
                pl.col(column).sum().alias(f"{column}_count")
                for column in ELIGIBILITY_COLUMNS
            ],
            pl.col("equity_eligible").sum().alias("equity_eligible_count"),
            member.sum().alias("member_count"),
            (member & (pl.col("source_assignment_type") == "DIRECT_ISIN"))
            .sum()
            .alias("direct_member_count"),
            (member & (pl.col("source_assignment_type") == "RECOVERED_RELABELED"))
            .sum()
            .alias("recovered_member_count"),
            (pl.col("action") == "ENTER").sum().alias("entries"),
            (pl.col("action") == "STAY").sum().alias("stays"),
            (pl.col("action") == "EXIT").sum().alias("exits"),
            (pl.col("action") == "OUT").sum().alias("outs"),
            pl.col("market_turnover_brl").first(),
            pl.col("turnover_brl").sum().alias("accepted_turnover_brl"),
            pl.col("turnover_brl").filter(member).sum().alias("member_turnover_brl"),
            pl.col("market_share").sum().alias("accepted_market_share_sum"),
            pl.col("market_share")
            .filter(member)
            .sum()
            .alias("member_market_share_sum"),
            pl.col("median_daily_turnover_brl")
            .filter(member)
            .min()
            .alias("member_median_turnover_min_brl"),
            pl.col("median_daily_turnover_brl")
            .filter(member)
            .median()
            .alias("member_median_turnover_median_brl"),
            pl.col("median_daily_turnover_brl")
            .filter(member)
            .max()
            .alias("member_median_turnover_max_brl"),
            pl.col("median_daily_trade_count")
            .filter(member)
            .min()
            .alias("member_median_trade_count_min"),
            pl.col("median_daily_trade_count")
            .filter(member)
            .median()
            .alias("member_median_trade_count_median"),
            pl.col("median_daily_trade_count")
            .filter(member)
            .max()
            .alias("member_median_trade_count_max"),
        )
        .with_columns(
            (pl.col("entries") + pl.col("exits")).alias("gross_churn_count"),
            (pl.col("accepted_turnover_brl") / pl.col("market_turnover_brl")).alias(
                "accepted_market_turnover_coverage"
            ),
            (pl.col("member_turnover_brl") / pl.col("market_turnover_brl")).alias(
                "member_market_turnover_coverage"
            ),
        )
        .sort("effective_from")
    )


def build_universe_tables(
    daily: pl.DataFrame,
    observations: pl.DataFrame,
    assignments: pl.DataFrame,
    config: UniverseConfig = CANONICAL_CONFIG,
) -> tuple[dict[str, pl.DataFrame], dict[str, object]]:
    daily = validate_daily(daily)
    assignments = assignments.sort("security_id")
    calendar = sorted(daily["trade_date"].unique().to_list())
    if not calendar:
        raise ValueError("COTAHIST daily input is empty")
    start_date = next(
        (trade_date for trade_date in calendar if trade_date >= config.start_date),
        None,
    )
    end_dates = [trade_date for trade_date in calendar if trade_date <= config.end_date]
    if start_date is None or not end_dates:
        raise ValueError("Research interval does not overlap the market calendar")
    resolved_end_date = end_dates[-1]
    valid_daily = daily.filter(valid_observation_expr())
    invalid_observation_count = daily.height - valid_daily.height
    accepted_ids = assignments["security_id"].to_list()
    accepted_valid = valid_daily.filter(pl.col("security_id").is_in(accepted_ids))
    observed_ids = set(accepted_valid["security_id"].unique().to_list())
    missing_ids = sorted(set(accepted_ids) - observed_ids)
    if missing_ids:
        raise ValueError(
            f"Accepted identities without valid COTAHIST observations: {missing_ids}"
        )
    first_seen = dict(
        accepted_valid.group_by("security_id")
        .agg(pl.col("trade_date").min())
        .iter_rows()
    )
    schedule = review_schedule(calendar, config)
    next_effective = {
        effective_from: (schedule[index + 1][1] if index + 1 < len(schedule) else None)
        for index, (_, effective_from) in enumerate(schedule)
    }
    anchor_candidates = [
        effective_from for _, effective_from in schedule if effective_from <= start_date
    ]
    anchor_effective = max(anchor_candidates) if anchor_candidates else schedule[0][1]

    prior_members: set[str] = set()
    rows: list[dict[str, object]] = []
    for review_date, effective_from in schedule:
        review_rows = _review_records(
            valid_daily=valid_daily,
            assignments=assignments,
            calendar=calendar,
            first_seen=first_seen,
            review_date=review_date,
            effective_from=effective_from,
            effective_to_exclusive=next_effective[effective_from],
            prior_members=prior_members,
            config=config,
        )
        prior_members = {
            str(row["security_id"]) for row in review_rows if row["is_member"]
        }
        if effective_from >= anchor_effective:
            rows.extend(review_rows)

    metrics = (
        pl.DataFrame(rows, infer_schema_length=None)
        .with_columns(
            pl.col("review_date").cast(pl.Date),
            pl.col("effective_from").cast(pl.Date),
            pl.col("effective_to_exclusive").cast(pl.Date),
            pl.col("window_start").cast(pl.Date),
            pl.col("window_end").cast(pl.Date),
            pl.col("last_observed_date").cast(pl.Date),
        )
        .sort(["effective_from", "security_id"])
    )
    membership = metrics.filter(pl.col("is_member"))
    changes = metrics.filter(pl.col("action").is_in(["ENTER", "EXIT"]))
    summary = _summary(metrics)
    segments = ticker_segments(observations)
    master = security_master(daily, segments)
    union = (
        membership.group_by("security_id")
        .agg(
            pl.col("effective_from").min().alias("first_effective_from"),
            pl.col("effective_from").max().alias("last_effective_from"),
            pl.col("ticker_asof").last().alias("latest_member_ticker"),
            pl.col("issuer_short_name_asof").last().alias("latest_member_issuer"),
            pl.col("effective_from").n_unique().alias("membership_months"),
            pl.col("market_share").median().alias("median_market_share_while_member"),
            pl.col("median_daily_turnover_brl")
            .median()
            .alias("median_daily_turnover_brl_while_member"),
            pl.col("median_daily_trade_count")
            .median()
            .alias("median_daily_trade_count_while_member"),
            pl.col("source_assignment_type").first(),
        )
        .join(master, on="security_id", how="left")
        .sort(["first_effective_from", "security_id"])
    )
    tables = {
        "universe_metrics_monthly": metrics,
        "universe_membership_monthly": membership,
        "universe_changes": changes,
        "universe_union": union,
        "security_master": master,
        "ticker_history": segments,
        "universe_summary": summary,
    }
    metadata = {
        "resolved_start_date": start_date,
        "resolved_end_date": resolved_end_date,
        "anchor_effective_date": anchor_effective,
        "market_sessions": len(calendar),
        "review_count": summary.height,
        "parent_security_count": master.height,
        "accepted_identity_count": len(accepted_ids),
        "union_security_count": union.height,
        "latest_member_count": int(summary["member_count"][-1]),
        "invalid_observation_count": invalid_observation_count,
    }
    return tables, metadata
