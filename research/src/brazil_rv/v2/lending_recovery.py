"""Admit reconciled BDI observations without regenerating model features."""

from __future__ import annotations

import numpy as np
import polars as pl


def prior_loan_references(quotes, dates, isins):
    """Previous published average, or last available, on the same ISIN only.

    B3's loan contract permits the last available average. This is a contractual
    reference, never an observed model price or label endpoint. Keep its source
    date, never use a current-session quote, and do not invent an adjustment from
    a successor's price. Event-specific reference overrides need separate terms.
    """
    dates = np.asarray(dates, dtype="datetime64[D]")
    values = np.full((len(dates), len(isins)), np.nan)
    published = np.full(values.shape, np.datetime64("NaT", "D"))
    names = {str(isin): index for index, isin in enumerate(isins)}
    quotes = quotes.filter(pl.col("isin").is_in(list(names)))
    if quotes.select(pl.struct("trade_date", "isin").is_duplicated().any()).item():
        raise ValueError("ambiguous published loan-reference quote")
    if quotes.filter(
        pl.col("average_brl").is_null()
        | ~pl.col("average_brl").is_finite()
        | (pl.col("average_brl") <= 0)
    ).height:
        raise ValueError("invalid published loan-reference quote")
    for (isin,), frame in quotes.partition_by("isin", as_dict=True).items():
        frame = frame.sort("trade_date")
        quote_dates = frame["trade_date"].to_numpy().astype("datetime64[D]")
        prior = np.searchsorted(quote_dates, dates, side="left") - 1
        has_prior = prior >= 0
        name = names[isin]
        values[has_prior, name] = frame["average_brl"].to_numpy()[prior[has_prior]]
        published[has_prior, name] = quote_dates[prior[has_prior]]
    return values, published


def _next_publication(frame, column, dates):
    source = frame[column].to_numpy().astype("datetime64[D]")
    positions = np.searchsorted(dates, source, side="right")
    inside = positions < len(dates)
    return frame.filter(pl.Series(inside)).with_columns(
        pl.Series("available_date", dates[positions[inside]])
    )


def recover_rates(accepted, recovered, dates):
    """Reconciled positive flow replaces its old key, including old parse errors.

    A printed zero-flow rate never refreshes a loan-price observation. Annual
    taker/donor decimals are already quantity-weighted from the audited rows;
    neither their difference nor a minimum is interpreted as our broker quote.
    """
    positive = recovered.filter(pl.col("registered_quantity") > 0)
    for field in ("annual_taker_rate", "annual_donor_rate"):
        if positive.filter(~pl.col(field).is_finite() | (pl.col(field) < 0)).height:
            raise ValueError("invalid positive-flow annual rate")
    new = _next_publication(positive, "source_trade_date", dates)
    keys = ["source_trade_date", "security_id"]
    if new.select(pl.struct(keys).is_duplicated().any()).item():
        raise ValueError("ambiguous recovered rate key")
    remaining = accepted.join(new.select(keys), on=keys, how="anti")
    return pl.concat([remaining, new], how="diagonal_relaxed").sort(
        "available_date", "security_id"
    )


def recover_balances(accepted, recovered, dates):
    """Keep only source-identified rows; preserve every previously admitted row.

    The report date controls availability, even when the balance date is older.
    Missing historical identity stays in the audit quarantine, never assigned
    from a later ticker. No rate or executable locate is inferred from quantity.
    """
    if recovered.filter(
        pl.col("quote_isin").is_not_null()
        & pl.col("isin").is_not_null()
        & (pl.col("quote_isin") != pl.col("isin"))
    ).height:
        raise ValueError("source and contemporaneous quote identities conflict")
    identified = recovered.filter(pl.col("isin").is_not_null())
    if identified.filter(
        (pl.col("position_date") > pl.col("report_date"))
        | (pl.col("quantity") < 0)
        | ~pl.col("balance_brl").is_finite()
        | (pl.col("balance_brl") < 0)
    ).height:
        raise ValueError("invalid identified source balance")
    new = _next_publication(identified, "report_date", dates).select(
        pl.col("position_date").alias("source_position_date"),
        pl.col("report_date").alias("source_report_date"),
        "available_date",
        (pl.lit("ISIN:") + pl.col("isin")).alias("security_id"),
        pl.when(pl.col("quote_isin").is_not_null())
        .then(pl.lit("same_date_cotahist_ticker"))
        .otherwise(pl.lit("printed_isin"))
        .alias("source_identity_method"),
        pl.col("quantity").alias("lending_balance_quantity"),
        pl.col("balance_brl").alias("lending_balance_brl"),
    )
    keys = ["source_report_date", "security_id"]
    if new.select(pl.struct(keys).is_duplicated().any()).item():
        raise ValueError("ambiguous recovered balance key")
    overlap = accepted.join(new, on=keys, suffix="_new")
    for field in (
        "source_position_date",
        "lending_balance_quantity",
        "lending_balance_brl",
    ):
        if overlap.filter(pl.col(field) != pl.col(field + "_new")).height:
            raise ValueError("balance overlap requires explicit source adjudication")
    additions = new.join(accepted.select(keys), on=keys, how="anti")
    return pl.concat([accepted, additions], how="vertical_relaxed").sort(
        "available_date", "security_id"
    )
