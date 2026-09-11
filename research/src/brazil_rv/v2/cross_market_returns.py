"""Same-endpoint ADR/EWZ return gaps and the registered Brent clock."""

from __future__ import annotations

from datetime import UTC, date, datetime, time

import numpy as np
import polars as pl

from .round5_market import SAO_PAULO


def admit_brent(levels: pl.DataFrame, sessions: list[date]) -> pl.DataFrame:
    """Day-t assessed Brent is known at the next B3 decision, never same-day."""
    clocks = {}
    for day in levels.filter(pl.col("series") == "brent_spot")["reference_date"]:
        position = int(np.searchsorted(sessions, day, side="right"))
        clocks[day] = (
            datetime.combine(sessions[position], time(15, 45), SAO_PAULO).astimezone(
                UTC
            )
            if position < len(sessions)
            else None
        )
    return levels.with_columns(
        pl.when(pl.col("series") == "brent_spot")
        .then(
            pl.col("reference_date").replace_strict(
                clocks, default=None, return_dtype=pl.Datetime("us", "UTC")
            )
        )
        .otherwise(pl.col("available_at"))
        .alias("available_at")
    )


def relative_return_panel(
    us_returns: pl.DataFrame,
    ptax: pl.DataFrame,
    cash: pl.DataFrame,
    pairs: list[dict],
    sessions: list[date],
    isins: tuple[str, ...],
    equity_returns: np.ndarray,
    equity_valid: np.ndarray,
    bova_returns: np.ndarray,
):
    """Compare adjacent observed closes with identical reference endpoints.

    An ADR in BRL earns its USD log return plus the exact same-endpoint PTAX
    log return. Subtract the B3 shareholder-wealth log return. US and B3 holiday
    mismatches stay missing; they are never reinterpreted as one-day returns.
    Dated ticker/class segments must hold at both endpoints and resolve to the
    same ISIN. No return crosses a share-class conversion. The listed flag means
    membership in the explicitly covered ADR-program roster, not all world ADRs.
    """
    shape = (len(sessions), len(isins))
    gap, gap_valid = np.zeros(shape, np.float32), np.zeros(shape, bool)
    flag = np.zeros(shape, np.float32)
    ewz, ewz_valid = np.zeros(len(sessions), np.float32), np.zeros(len(sessions), bool)
    lookup = {isin: i for i, isin in enumerate(isins)}
    ticker_set = {p["ticker"] for p in pairs}
    assignments = {}
    for row in (
        cash.filter(pl.col("ticker").is_in(ticker_set))
        .select("source_trade_date", "ticker", "isin")
        .unique()
        .iter_rows(named=True)
    ):
        key = row["source_trade_date"], row["ticker"]
        if key in assignments and assignments[key] != row["isin"]:
            raise ValueError("ADR identity has multiple ISINs at one dated ticker")
        assignments[key] = row["isin"]
    returns = {
        (r["series"], r["reference_date"]): r for r in us_returns.iter_rows(named=True)
    }
    fx = {
        r["reference_date"]: r
        for r in ptax.filter(pl.col("series") == "ptax_brl_per_usd").iter_rows(
            named=True
        )
    }
    parsed = [
        {
            **p,
            "start": date.fromisoformat(p["start"]),
            "end": date.fromisoformat(p["end"]),
        }
        for p in pairs
    ]
    last_identity = {}
    for i in range(1, len(sessions)):
        reference = sessions[i - 1]
        for ticker in ticker_set:
            if (reference, ticker) in assignments:
                last_identity[ticker] = assignments[reference, ticker]
        cutoff = datetime.combine(sessions[i], time(15, 45), SAO_PAULO).astimezone(UTC)
        previous = sessions[i - 2] if i >= 2 else None
        for pair in parsed:
            if not pair["start"] <= reference <= pair["end"]:
                continue
            name = lookup.get(last_identity.get(pair["ticker"]))
            if name is not None:
                flag[i, name] = 1
            now_id = assignments.get((reference, pair["ticker"]))
            before_id = assignments.get((previous, pair["ticker"]))
            record = returns.get(("us_" + pair["symbol"], reference))
            current_fx, previous_fx = fx.get(reference), fx.get(previous)
            if (
                name is None
                or previous is None
                or previous < pair["start"]
                or now_id != isins[name]
                or before_id != now_id
                or record is None
                or record["previous_date"] != previous
                or record["available_at"] is None
                or record["available_at"] > cutoff
                or not equity_valid[i - 1, name]
                or any(
                    r is None
                    or r["available_at"] is None
                    or r["available_at"] > cutoff
                    or r["value"] <= 0
                    for r in (current_fx, previous_fx)
                )
            ):
                continue
            value = (
                record["log_return"]
                + np.log(current_fx["value"] / previous_fx["value"])
                - equity_returns[i - 1, name]
            )
            if np.isfinite(value):
                gap[i, name], gap_valid[i, name] = value, True
        record = returns.get(("us_EWZ", reference))
        current_fx, previous_fx = fx.get(reference), fx.get(previous)
        if (
            record is not None
            and record["previous_date"] == previous
            and record["available_at"] is not None
            and record["available_at"] <= cutoff
            and all(
                r is not None
                and r["available_at"] is not None
                and r["available_at"] <= cutoff
                and r["value"] > 0
                for r in (current_fx, previous_fx)
            )
            and np.isfinite(bova_returns[i - 1])
        ):
            ewz[i] = (
                record["log_return"]
                + np.log(current_fx["value"] / previous_fx["value"])
                - bova_returns[i - 1]
            )
            ewz_valid[i] = True
    return gap, gap_valid, ewz, ewz_valid, flag
