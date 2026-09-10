"""Bounded B3 source normalization for the registered Round-5 data audit."""

from __future__ import annotations

import json
import io
import re
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from xml.etree.ElementTree import iterparse
from zoneinfo import ZoneInfo

import polars as pl

from brazil_rv.v2.artifacts import sha256_file
from brazil_rv.preprocessing.b3_options_open_interest import (
    _descendant,
    _instrument_id,
    _local,
    _text,
)
from brazil_rv.preprocessing.options_activity import _choose_txt_member, parse_option_line


RATE_FIELDS = (
    "donor_min",
    "donor_avg",
    "donor_max",
    "taker_min",
    "taker_avg",
    "taker_max",
)


@contextmanager
def historical_xml(path: Path, available_date: date):
    """Stream the latest B3-created version known before the first decision.

    B3's published daily-report timetable is in Sao Paulo time. ZIP file order
    and the server's current HTTP modification time do not identify a vintage.
    """
    cutoff = datetime.combine(available_date, datetime.min.time()).replace(
        hour=15, minute=45, tzinfo=ZoneInfo("America/Sao_Paulo")
    )
    with zipfile.ZipFile(path) as outer:
        nested = [p for p in outer.infolist() if not p.is_dir()]
        if len(nested) != 1:
            raise ValueError("Expected exactly one B3 nested archive")
        payload = outer.read(nested[0])
    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
        versions = []
        for member in inner.infolist():
            if not member.filename.endswith(".xml"):
                continue
            with inner.open(member) as handle:
                header = handle.read(8192)
            match = re.search(rb"<(?:[\w]+:)?CreDtAndTm>([^<]+)", header)
            if match is None:
                raise ValueError(f"Missing creation time: {member.filename}")
            timestamp = datetime.fromisoformat(match[1].decode())
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=ZoneInfo("America/Sao_Paulo"))
            versions.append((timestamp, member.filename))
        known = [(stamp, member) for stamp, member in versions if stamp <= cutoff]
        if not known:
            raise ValueError("No historical XML vintage existed by the first decision")
        selected_time, selected_member = max(known)
        audit = {
            "selected_member": selected_member,
            "selected_creation_time": selected_time.isoformat(),
            "versions_after_decision_excluded": len(versions) - len(known),
            "versions": [
                {"creation_time": t.isoformat(), "member": m} for t, m in versions
            ],
        }
        with inner.open(selected_member) as handle:
            yield handle, audit


def parse_options_snapshot(
    in_path: Path,
    pr_path: Path,
    source_date: date,
    available_date: date,
    identities: dict[str, tuple[date, date]],
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    """Extract reported OI and nonregular trading without inventing missing fields."""
    cash = {}
    options = {}
    with historical_xml(in_path, available_date) as (handle, in_audit):
        for _, element in iterparse(handle, events=("end",)):
            if _local(element.tag) != "Instrm":
                continue
            identifier = _instrument_id(element)
            info = _descendant(element, "InstrmInf")
            equity = _descendant(info, "EqtyInf")
            option = _descendant(info, "OptnOnEqtsInf")
            active = _text(element, "ActvtyInd")
            if (
                active == "true"
                and equity is not None
                and _text(element, "Mkt") == "10"
            ):
                isin = _text(equity, "ISIN")
                bounds = identities.get(isin)
                if bounds and bounds[0] <= source_date <= bounds[1]:
                    cash[identifier] = isin
            elif active == "true" and option is not None:
                start = _text(option, "TradgStartDt")
                end = _text(option, "TradgEndDt")
                expiry = _text(option, "XprtnDt")
                if (
                    start
                    and end
                    and expiry
                    and start <= str(source_date) <= min(end, expiry)
                ):
                    options[identifier] = (
                        _text(_descendant(option, "UndrlygInstrmId"), "Id"),
                        _text(option, "OptnTp"),
                        _text(option, "TckrSymb"),
                    )
            element.clear()
    options = {key: value for key, value in options.items() if value[0] in cash}
    aggregates = defaultdict(
        lambda: {
            "listed_series": 0,
            "reported_series": 0,
            "oi_observed_series": 0,
            "call_oi": 0.0,
            "put_oi": 0.0,
        }
    )
    for underlying, _, _ in options.values():
        aggregates[cash[underlying]]["listed_series"] += 1
    stock_rows = []
    seen = set()
    with historical_xml(pr_path, available_date) as (handle, pr_audit):
        for _, element in iterparse(handle, events=("end",)):
            if _local(element.tag) != "PricRpt":
                continue
            identifier = _instrument_id(element)
            if identifier in options or identifier in cash:
                if _text(_descendant(element, "TradDt"), "Dt") != str(source_date):
                    raise ValueError(
                        "Price-report date does not match its archive date"
                    )
                if identifier in seen:
                    raise ValueError("Duplicate instrument in daily price report")
                seen.add(identifier)
            if identifier in options:
                underlying, kind, ticker = options[identifier]
                if _text(element, "TckrSymb") != ticker:
                    raise ValueError(
                        "Instrument ID changed ticker within its daily pair"
                    )
                agg = aggregates[cash[underlying]]
                agg["reported_series"] += 1
                oi = _text(element, "OpnIntrst")
                if oi is not None:
                    value = float(oi)
                    if value < 0 or kind not in ("CALL", "PUT", "PUTT"):
                        raise ValueError("Invalid option OI or option type")
                    agg["oi_observed_series"] += 1
                    agg["call_oi" if kind == "CALL" else "put_oi"] += value
            if identifier in cash:
                row = {"isin": cash[identifier]}
                for name, tag in (
                    ("quantity", "FinInstrmQty"),
                    ("trades", "TradQty"),
                    ("volume_brl", "NtlFinVol"),
                    ("regular_quantity", "RglrTraddCtrcts"),
                    ("nonregular_quantity", "NonRglrTraddCtrcts"),
                ):
                    raw = _text(element, tag)
                    row[name] = float(raw) if raw is not None else None
                stock_rows.append(row)
            element.clear()
    oi_rows = [{"isin": isin, **values} for isin, values in aggregates.items()]
    for row in oi_rows:
        row["oi_all_listed_observed"] = (
            row["oi_observed_series"] == row["listed_series"]
        )
    return (
        pl.DataFrame(oi_rows),
        pl.DataFrame(stock_rows),
        {
            "source_trade_date": str(source_date),
            "available_date": str(available_date),
            "in": in_audit,
            "pr": pr_audit,
            "mapped_active_series": len(options),
            "stock_instruments": len(cash),
        },
    )


def cotahist_option_quantities(
    archive_path: Path,
    identities: dict[str, tuple[date, date]],
    *,
    end: date,
) -> pl.DataFrame:
    """Aggregate traded units using COTAHIST's explicit underlying cash ISIN.

    Pre-BVBG listing-only zero days are unknown. Rows exist only when an option
    trade was actually printed; a missing stock-day is not fabricated as zero.
    """
    totals = defaultdict(lambda: [0, 0, 0])
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(_choose_txt_member(archive, archive_path)) as handle:
            for raw in handle:
                option = parse_option_line(raw.rstrip(b"\r\n"))
                if option is None:
                    continue
                day = option.trade_date
                bounds = identities.get(option.isin)
                if day > end or bounds is None or not bounds[0] <= day <= bounds[1]:
                    continue
                values = totals[(day, option.isin)]
                values[int(option.is_put)] += option.quantity
                values[2] += 1
    return pl.DataFrame(
        [
            {
                "source_trade_date": day,
                "isin": isin,
                "call_quantity": v[0],
                "put_quantity": v[1],
                "traded_option_series": v[2],
            }
            for (day, isin), v in sorted(totals.items())
        ]
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
