"""Bounded B3 source normalization for the registered Round-5 data audit."""

from __future__ import annotations

import json
import io
import re
import zipfile
from bisect import bisect_right
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from xml.etree.ElementTree import iterparse
from zoneinfo import ZoneInfo

import polars as pl
import numpy as np

from brazil_rv.v2.artifacts import sha256_file
from brazil_rv.preprocessing.b3_options_open_interest import (
    _descendant,
    _instrument_id,
    _local,
    _text,
)
from brazil_rv.preprocessing.options_activity import (
    _choose_txt_member,
    parse_option_line,
)


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


def stitch_legacy_balances(
    old: pl.DataFrame,
    extracted: pl.DataFrame,
    cash: pl.DataFrame,
    admissible_reports: list[date],
    sessions: list[date],
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Bind printed legacy tickers to exact dated cash identities, without carry."""
    legacy = extracted.filter(pl.col("report_date").is_in(admissible_reports))
    mapping = cash.select("source_trade_date", "ticker", "isin").unique()
    unambiguous = (
        mapping.group_by("source_trade_date", "ticker")
        .agg(pl.col("isin").n_unique().alias("identity_count"), pl.col("isin").first())
        .filter(pl.col("identity_count") == 1)
        .drop("identity_count")
    )
    joined = legacy.drop("isin").join(
        unambiguous,
        left_on=["position_date", "ticker"],
        right_on=["source_trade_date", "ticker"],
        how="inner",
    )
    next_sessions = pl.DataFrame(
        {"report_date": sessions[:-1], "available_date": sessions[1:]},
        schema={"report_date": pl.Date, "available_date": pl.Date},
    )
    new = (
        joined.join(next_sessions, on="report_date", how="inner")
        .select(
            pl.col("position_date").alias("source_position_date"),
            pl.col("report_date").alias("source_report_date"),
            "available_date",
            (pl.lit("ISIN:") + pl.col("isin")).alias("security_id"),
            pl.lit("exact_position_date_cotahist_ticker_isin").alias(
                "source_identity_method"
            ),
            pl.col("quantity").alias("lending_balance_quantity"),
            pl.col("balance_brl").alias("lending_balance_brl"),
        )
        .cast(old.schema)
    )
    keys = ["available_date", "security_id"]
    additions = new.join(old.select(keys), on=keys, how="anti")
    result = pl.concat([old, additions]).sort(keys)
    if result.select(pl.struct(keys).is_duplicated().any()).item():
        raise ValueError("Legacy lending identities collide at one publication")
    return result, {
        "old_rows_preserved_exactly": old.height,
        "admissible_raw_rows": legacy.height,
        "exact_identity_matches": joined.height,
        "identity_unresolved_rows": legacy.height - joined.height,
        "added_rows": additions.height,
        "added_isins": additions["security_id"].n_unique(),
        "first_added_available_date": str(additions["available_date"].min()),
        "last_added_available_date": str(additions["available_date"].max()),
    }


def lending_decision_features(
    balances: pl.DataFrame,
    rates: pl.DataFrame,
    sessions: list[date],
    isins: list[str],
    daily_volume_brl: np.ndarray,
) -> pl.DataFrame:
    """Extend the existing five source-date/vintage formulas without redefining them."""
    from brazil_rv.v2.sidecars import _raw_lending_features

    def identity(frame):
        return frame.with_columns(
            pl.col("security_id").str.strip_prefix("ISIN:").alias("isin")
        ).drop("security_id")

    raw = identity(balances).join(
        identity(rates), on=["available_date", "isin"], how="full", coalesce=True
    )
    result = _raw_lending_features(raw, sessions, isins, daily_volume_brl)
    positions = {day: index for index, day in enumerate(sessions)}
    fields = [
        "loan_balance_to_volume_20",
        "loan_balance_change_1",
        "loan_balance_change_5",
        "loan_rate",
        "loan_rate_change_5",
    ]
    columns = [pl.col("available_date").alias("date"), pl.col("isin")]
    for feature in fields:
        source = (
            "source_trade_date"
            if feature.startswith("loan_rate")
            else "source_position_date"
        )
        age = pl.col("available_date").replace_strict(
            positions, default=None, return_dtype=pl.Int32
        ) - pl.col(source).replace_strict(
            positions, default=None, return_dtype=pl.Int32
        )
        columns.extend(
            [
                pl.when(pl.col(feature + "_mask")).then(pl.col(feature)).alias(feature),
                pl.when(pl.col(feature + "_mask"))
                .then(age)
                .alias(feature + "_age_sessions"),
            ]
        )
    features = result.select(columns)
    rate_grid = (
        pl.DataFrame({"source_trade_date": sessions[:-1], "date": sessions[1:]})
        .join(pl.DataFrame({"isin": isins}), how="cross")
        .join(
            identity(rates).select("source_trade_date", "isin", "registered_quantity"),
            on=["source_trade_date", "isin"],
            how="left",
        )
        .sort("isin", "source_trade_date")
    )
    rate_grid = (
        rate_grid.with_columns(
            pl.col("registered_quantity")
            .shift(1)
            .rolling_mean(window_size=20, min_samples=15)
            .over("isin")
            .alias("prior_mean"),
            pl.col("registered_quantity")
            .shift(1)
            .rolling_std(window_size=20, min_samples=15, ddof=1)
            .over("isin")
            .alias("prior_std"),
        )
        .with_columns(
            pl.when(
                (pl.col("prior_std") > 0)
                & pl.col("prior_std").is_finite()
                & (pl.col("registered_quantity") > 0)
                & (pl.col("source_trade_date").replace_strict(positions) >= 20)
            )
            .then(
                (pl.col("registered_quantity") - pl.col("prior_mean"))
                / pl.col("prior_std")
            )
            .alias("new_loan_volume_surprise")
        )
        .filter(pl.col("new_loan_volume_surprise").is_not_null())
        .select(
            "date",
            "isin",
            "new_loan_volume_surprise",
            pl.lit(1).alias("new_loan_volume_surprise_age_sessions"),
        )
    )
    return (
        features.join(rate_grid, on=["date", "isin"], how="full", coalesce=True)
        .with_columns(
            pl.lit(None, dtype=pl.Float64).alias("utilization_proxy"),
            pl.lit(None, dtype=pl.Int32).alias("utilization_proxy_age_sessions"),
        )
        .sort("date", "isin")
    )


def lending_utilization_features(
    balances: pl.DataFrame,
    free_float: pl.DataFrame,
    identity: pl.DataFrame,
    sessions: list[date],
    market: dict,
    capital_changes: list[dict],
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Loan shares / known, dated freely circulating shares of the same class.

    The FRE filing year is not the float measurement date. Its last-meeting
    snapshot date governs quantity-unit continuity, while receipt governs
    knowledge. An observed unit boundary or known intervening capital change
    invalidates the old denominator; shares are never retrospectively adjusted.
    """
    positions = {day: index for index, day in enumerate(sessions)}
    mapped = identity.with_columns(pl.col("cnpj").str.slice(0, 8).alias("issuer_root"))
    class_counts = mapped.group_by("date", "issuer_root", "cvm_code", "class").agg(
        pl.col("isin").n_unique().alias("class_security_count")
    )
    mapped = mapped.join(
        class_counts, on=["date", "issuer_root", "cvm_code", "class"]
    )
    joined = (
        balances.with_columns(
            pl.col("available_date").alias("date"),
            pl.col("security_id").str.strip_prefix("ISIN:").alias("isin"),
        )
        .join(mapped, on=["date", "isin"], how="left")
        .sort("date", "isin")
    )
    sources = defaultdict(list)
    for row in free_float.iter_rows(named=True):
        if row["snapshot_date"] is not None and row["free_float_shares"] > 0:
            sources[(row["cnpj"][:8], row["cvm_code"], row["class"])].append(row)
    for rows in sources.values():
        rows.sort(key=lambda r: (r["date"], r["version"], r["document_id"]))
    changes = defaultdict(list)
    for event in capital_changes:
        changes[(event["cnpj"][:8], event["cvm_code"])].append(event)
    cursor = defaultdict(int)
    known = defaultdict(list)
    audit = defaultdict(int)
    output = []
    for row in joined.iter_rows(named=True):
        index = positions[row["date"]]
        source_index = positions.get(row["source_position_date"])
        key = (row["issuer_root"], row["cvm_code"], row["class"])
        source = sources[key]
        while cursor[key] < len(source) and source[cursor[key]]["date"] <= row["date"]:
            known[key].append(source[cursor[key]])
            cursor[key] += 1
        if (
            row["class"] not in {"ON", "PN"}
            or row["class_security_count"] != 1
            or row["identity_effective_start"] > row["source_position_date"]
        ):
            audit["identity_or_class_unavailable"] += 1
            continue
        candidates = [
            d for d in known[key] if d["snapshot_date"] <= row["source_position_date"]
        ]
        if not candidates:
            audit["dated_float_unavailable"] += 1
            continue
        document = max(
            candidates,
            key=lambda d: (
                d["snapshot_date"], d["reference"], d["version"], d["date"], d["document_id"]
            ),
        )
        if any(
            event["available_index"] <= index
            and document["snapshot_date"] < event["effective"] <= row["source_position_date"]
            for event in changes[(row["issuer_root"], row["cvm_code"])]
        ):
            audit["known_capital_boundary"] += 1
            continue
        column = market["columns"].get(row["isin"])
        first = bisect_right(sessions, document["snapshot_date"])
        prefix = market["barrier_prefix"]
        if (
            column is None
            or source_index is None
            or first > source_index + 1
            or prefix[source_index + 1, column] != prefix[first, column]
        ):
            audit["observed_unit_boundary"] += 1
            continue
        quantity = row["lending_balance_quantity"]
        if quantity is None or quantity < 0:
            audit["loan_quantity_unavailable"] += 1
            continue
        output.append(
            {
                "date": row["date"],
                "isin": row["isin"],
                "utilization_proxy": quantity / document["free_float_shares"],
                "utilization_proxy_age_sessions": index
                - min(source_index, positions[document["date"]]),
            }
        )
    audit["source_balance_rows"] = balances.height
    audit["valid_rows"] = len(output)
    return pl.DataFrame(
        output,
        schema={
            "date": pl.Date,
            "isin": pl.String,
            "utilization_proxy": pl.Float64,
            "utilization_proxy_age_sessions": pl.Int32,
        },
    ), dict(audit)


def activity_decision_features(
    cash: pl.DataFrame,
    option_quantities: pl.DataFrame,
    snapshots: pl.DataFrame,
    nonregular: pl.DataFrame,
    sessions: list[date],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Causal trailing option activity and mean trade size on the fixed calendar."""
    dates = pl.DataFrame({"source_trade_date": sessions[:-1], "date": sessions[1:]})
    grid = (
        dates.join(cash.select("isin").unique(), how="cross")
        .join(
            cash.select(
                "source_trade_date", "isin", "quantity", "volume_brl", "trades"
            ),
            on=["source_trade_date", "isin"],
            how="left",
        )
        .join(option_quantities, on=["source_trade_date", "isin"], how="left")
    )
    if snapshots.height:
        grid = grid.join(
            snapshots.select(
                "source_trade_date",
                "isin",
                "listed_series",
                "call_oi",
                "put_oi",
                "oi_all_listed_observed",
            ),
            on=["source_trade_date", "isin"],
            how="left",
        )
    else:
        grid = grid.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("listed_series"),
            pl.lit(None, dtype=pl.Float64).alias("call_oi"),
            pl.lit(None, dtype=pl.Float64).alias("put_oi"),
            pl.lit(False).alias("oi_all_listed_observed"),
        )
    # Complete COTAHIST reports all executed trades. A known listed series with
    # no printed trade has zero activity; absent historical listing status stays unknown.
    grid = grid.with_columns(
        [
            pl.when(pl.col("listed_series") > 0)
            .then(pl.col(name).fill_null(0))
            .otherwise(pl.col(name))
            .cast(pl.Float64)
            .alias(name)
            for name in ("call_quantity", "put_quantity")
        ]
    ).sort("isin", "source_trade_date")

    def rolling(name, window):
        return (
            pl.col(name)
            .rolling_sum(window_size=window, min_samples=window)
            .over("isin")
        )

    grid = (
        grid.with_columns(
            (pl.col("call_quantity") + pl.col("put_quantity")).alias("option_quantity"),
            pl.when(pl.col("oi_all_listed_observed"))
            .then(pl.col("call_oi") + pl.col("put_oi"))
            .alias("total_oi"),
        )
        .with_columns(
            rolling("quantity", 20).alias("stock_quantity_20"),
            # BVBG.086 OpnIntrst is the opening D / closing D-1 position.
            # The end-of-day PR publication is consumed at D+1, age two.
            pl.col("quantity")
            .shift(1)
            .rolling_sum(window_size=20, min_samples=20)
            .over("isin")
            .alias("oi_stock_quantity_20"),
            rolling("option_quantity", 20).alias("option_quantity_20"),
            rolling("put_quantity", 5).alias("put_quantity_5"),
            rolling("call_quantity", 5).alias("call_quantity_5"),
            rolling("volume_brl", 20).alias("stock_brl_20"),
            rolling("trades", 20).alias("stock_trades_20"),
            pl.col("total_oi").diff().over("isin").alias("oi_change"),
        )
        .with_columns(
            pl.when(pl.col("stock_quantity_20") > 0)
            .then(pl.col("option_quantity_20") / pl.col("stock_quantity_20"))
            .alias("option_to_stock_volume_20"),
            pl.when(pl.col("call_quantity_5") > 0)
            .then(pl.col("put_quantity_5") / pl.col("call_quantity_5"))
            .alias("put_call_volume_ratio_5"),
            pl.when(pl.col("oi_all_listed_observed"))
            .then(((pl.col("put_oi") + 1) / (pl.col("call_oi") + 1)).log())
            .alias("put_call_oi_log_ratio"),
            pl.when(pl.col("oi_stock_quantity_20") > 0)
            .then(pl.col("oi_change") / (pl.col("oi_stock_quantity_20") / 20))
            .alias("delta_oi_to_volume_1"),
            pl.lit(None, dtype=pl.Float64).alias("uncovered_call_share"),
            pl.when(pl.col("stock_trades_20") > 0)
            .then(pl.col("stock_brl_20") / pl.col("stock_trades_20"))
            .alias("avg_trade_size_20"),
        )
    )
    if nonregular.height:
        after = nonregular.select(
            "source_trade_date",
            "isin",
            pl.col("quantity").alias("reported_quantity"),
            pl.when(pl.col("nonregular_quantity").is_not_null())
            .then(pl.col("nonregular_quantity"))
            .otherwise(pl.col("quantity") - pl.col("regular_quantity"))
            .alias("after_quantity"),
        )
        grid = grid.join(after, on=["source_trade_date", "isin"], how="left").sort(
            "isin", "source_trade_date"
        )
        grid = grid.with_columns(
            pl.when(
                (pl.col("after_quantity") >= 0)
                & (pl.col("after_quantity") <= pl.col("reported_quantity"))
            )
            .then(pl.col("after_quantity"))
            .alias("after_quantity")
        )
        grid = grid.with_columns(
            rolling("after_quantity", 5).alias("after_quantity_5"),
            rolling("reported_quantity", 5).alias("reported_quantity_5"),
        )
        grid = grid.with_columns(
            pl.when(pl.col("reported_quantity_5") > 0)
            .then(pl.col("after_quantity_5") / pl.col("reported_quantity_5"))
            .alias("after_hours_volume_share_5")
        )
    else:
        grid = grid.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("after_hours_volume_share_5")
        )

    def selected(features):
        return (
            grid.select(
                "date",
                "isin",
                *features,
                *[
                    pl.when(pl.col(feature).is_not_null())
                    .then(
                        2
                        if feature in {"put_call_oi_log_ratio", "delta_oi_to_volume_1"}
                        else 1
                    )
                    .alias(feature + "_age_sessions")
                    for feature in features
                ],
            )
            .filter(pl.any_horizontal(pl.col(name).is_not_null() for name in features))
            .sort("date", "isin")
        )

    return selected(
        [
            "option_to_stock_volume_20",
            "put_call_volume_ratio_5",
            "put_call_oi_log_ratio",
            "delta_oi_to_volume_1",
            "uncovered_call_share",
        ]
    ), selected(["avg_trade_size_20", "after_hours_volume_share_5"])
