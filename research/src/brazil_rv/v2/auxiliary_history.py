"""Bounded source-history views for admitted unit, no-cash equity renames.

These views change aggregate share history, not the assignment of a raw source
record or the identity of any option series/loan contract. Callers supply a
source-bound same-class link and publish only the returned, decision-gated rows.
"""

from datetime import date

import polars as pl

from .round5_b3 import activity_decision_features
from .sidecars import derive_known_archive_features


def _prior_history(source, predecessor, successor, effective):
    prior = source.filter(
        (pl.col("isin") == predecessor) & (pl.col("source_trade_date") < effective)
    ).with_columns(pl.lit(successor).alias("isin"))
    existing = source.filter(pl.col("isin") == successor)
    if prior.join(existing, on=["source_trade_date", "isin"], how="inner").height:
        raise ValueError("rename history overlaps an original successor observation")
    return pl.concat([source, prior], how="vertical_relaxed")


def renamed_activity_features(
    cash,
    quantities,
    snapshots,
    nonregular,
    sessions: list[date],
    *,
    predecessor: str,
    successor: str,
    effective: date,
    known: date,
):
    """Recompute only a bounded pair's original 20/5-session reducers.

    Original publication dates and full/observed-subset OI support are retained.
    In particular, an incomplete aggregate cannot acquire a full-OI delta from
    this history view. No option-series or loan alias is inferred.
    """
    sources = [
        _prior_history(frame, predecessor, successor, effective)
        for frame in (cash, quantities, snapshots, nonregular)
    ]
    results = activity_decision_features(*sources, sessions)
    return tuple(
        frame.filter(
            (pl.col("isin") == successor) & (pl.col("date") >= max(effective, known))
        )
        for frame in results
    )


def renamed_oddlot_features(
    source,
    sessions: list[date],
    *,
    predecessor: str,
    successor: str,
    effective: date,
    known: date,
):
    """Preserve publication-vintage checks and exact five-session BRL-share lag."""
    history = _prior_history(source, predecessor, successor, effective)
    return derive_known_archive_features(
        history, sessions, (predecessor, successor), group="oddlot"
    ).filter(
        (pl.col("isin") == successor)
        & (pl.col("available_date") >= max(effective, known))
    )
