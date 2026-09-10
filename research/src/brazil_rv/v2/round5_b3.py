"""Bounded B3 source normalization for the registered Round-5 data audit."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from brazil_rv.v2.artifacts import sha256_file


RATE_FIELDS = (
    "donor_min",
    "donor_avg",
    "donor_max",
    "taker_min",
    "taker_avg",
    "taker_max",
)


def stitch_registered_rates(
    old: pl.DataFrame,
    registered: pl.DataFrame,
    identities: pl.DataFrame,
    sessions: list[date],
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    """Keep old rates exact; admit new positive registered-flow date/ISIN keys.

    Printed donor/taker rates are annual percentage points. The economic rate is
    the quantity-weighted taker average divided by 100. Registered flow includes
    manual renewals, and zero-flow rows must not refresh the observation's age.
    D's registered activity is published at the next session's market open.
    """
    next_sessions = pl.DataFrame(
        {"report_date": sessions[:-1], "available_date": sessions[1:]},
        schema={"report_date": pl.Date, "available_date": pl.Date},
    )
    dated = registered.join(
        identities.select("isin", "first_date", "last_date"), on="isin", how="inner"
    ).filter(
        pl.col("report_date").is_between(pl.col("first_date"), pl.col("last_date"))
    )
    positive = dated.filter(pl.col("quantity") > 0).join(
        next_sessions, on="report_date", how="inner"
    )
    if positive.select(
        pl.any_horizontal(
            [~pl.col(field).is_finite() | (pl.col(field) < 0) for field in RATE_FIELDS]
        ).any()
    ).item():
        raise ValueError("Positive registered lending flow has invalid annual rates")
    raw = positive.drop("first_date", "last_date").sort("report_date", "isin", "ticker")
    new = (
        positive.group_by("report_date", "available_date", "isin")
        .agg(
            pl.col("contracts").sum().alias("registered_contracts"),
            pl.col("quantity").sum().alias("registered_quantity"),
            (
                (pl.col("taker_avg") * pl.col("quantity")).sum()
                / pl.col("quantity").sum()
                / 100.0
            ).alias("annual_taker_rate"),
        )
        .with_columns((pl.lit("ISIN:") + pl.col("isin")).alias("security_id"))
        .rename({"report_date": "source_trade_date"})
        .select(old.columns)
        .cast(old.schema)
    )
    keys = ["available_date", "security_id"]
    overlap = new.join(old, on=keys, suffix="_old")
    if (
        overlap.height
        and overlap.filter(
            (pl.col("registered_quantity") != pl.col("registered_quantity_old"))
            | (
                (pl.col("annual_taker_rate") - pl.col("annual_taker_rate_old")).abs()
                > 1e-12
            )
        ).height
    ):
        raise ValueError(
            "Reparsed lending overlaps differ beyond the invertible transform"
        )
    additions = new.join(old.select(keys), on=keys, how="anti")
    result = pl.concat([old, additions]).sort(
        "available_date", "source_trade_date", "security_id"
    )
    if result.select(pl.struct(keys).is_duplicated().any()).item():
        raise ValueError("Duplicate economic lending observation")
    return (
        result,
        raw,
        {
            "old_rows_preserved_exactly": old.height,
            "overlap_rows": overlap.height,
            "new_positive_observations": additions.height,
            "zero_flow_rows_excluded": dated.filter(pl.col("quantity") <= 0).height,
            "source_isins": raw.get_column("isin").n_unique(),
            "first_added_available_date": str(
                additions.get_column("available_date").min()
            ),
            "last_added_available_date": str(
                additions.get_column("available_date").max()
            ),
        },
    )


def write_rate_source_candidate(
    *,
    old_root: Path,
    extracted_root: Path,
    identities: pl.DataFrame,
    sessions: list[date],
    output_root: Path,
    code_commit: str,
) -> dict[str, object]:
    """Freeze the rate-only candidate; source-vintage admission is explicit."""
    output_root.mkdir(parents=True, exist_ok=False)
    old_path = old_root / "lending_rates.parquet"
    raw_path = extracted_root / "registered_loans.parquet"
    old_manifest = json.loads((old_root / "manifest.json").read_text())
    raw_manifest = json.loads((extracted_root / "manifest.json").read_text())
    for path, expected in (
        (old_path, old_manifest["artifacts"][old_path.name]["sha256"]),
        (raw_path, raw_manifest["artifacts"][raw_path.name]["sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"Source identity changed: {path}")
    rates, raw, audit = stitch_registered_rates(
        pl.read_parquet(old_path), pl.read_parquet(raw_path), identities, sessions
    )
    rates.write_parquet(output_root / "lending_rates.parquet", compression="zstd")
    raw.write_parquet(
        output_root / "positive_registered_loan_sources.parquet", compression="zstd"
    )
    manifest = {
        "schema": "BRAZIL_RV_ROUND5_LENDING_RATE_SOURCE_CANDIDATE_V1",
        "status": "complete_source_candidate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": code_commit,
        "admission": "latest_vintage_unknown_revision_share; historical archive PDFs regenerated April 2025",
        "availability_rule": "Registered trades on D published at next B3 market open; first next-session 15:45 decision",
        "availability_evidence": "https://www.b3.com.br/data/files/32/02/C0/25/391EA810E9C1AAA8AC094EA8/Glossario%20_%20Emprestimos_Registrados.pdf",
        "rate_rule": "Positive quantity only; quantity-weighted taker_avg annual percentage points / 100",
        "flow_semantics": "Registered lending flow includes new loans and manual renewals",
        "old_archive": {
            "path": str(old_root),
            "manifest_sha256": sha256_file(old_root / "manifest.json"),
        },
        "raw_extraction": {
            "path": str(extracted_root),
            "manifest_sha256": sha256_file(extracted_root / "manifest.json"),
        },
        "audit": audit,
        "artifacts": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in output_root.glob("*.parquet")
        },
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
