"""Round-7 source audits and the bounded repair of the sealed daily store."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from .artifacts import sha256_file, write_json_atomic
from .contract import DEVELOPMENT_END, FINETUNE_START, PRETRAIN_END
from .round5_cvm import normalized, rad_rows
from .round5_derived import bind
from .round5_store import align_family
from .store import open_store_for_samples, peak_rss_bytes

PROJECT = Path(__file__).resolve().parents[4]
NATIVE_FIELDS = (
    "earnings_yield_ttm",
    "book_to_market",
    "gross_profitability",
    "liabilities_to_assets",
    "accruals_to_assets",
    "revenue_growth_yoy",
    "sue",
    "log_market_cap",
    "earnings_negative_flag",
)
CONTINUATION_CODES = frozenset({"06", "07", "08"})
UNIT_WORDS = ("desdobramento", "grupamento", "bonificacao", "conversao", "conversoes")


def registered_sources():
    """Resolve canonical evidence pointers, never choose a timestamped directory."""
    accepted_path = PROJECT / "docs/v2_round6_inputs.json"
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    root = Path(accepted["store"]["root"])
    if sha256_file(root / "manifest.json") != accepted["store"]["manifest_sha256"]:
        raise ValueError("Round-6 store identity differs from accepted pointer")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["axes"]["date_end"] > DEVELOPMENT_END.isoformat():
        raise PermissionError("Round-7 source enters protected history")
    cvm_pointer = PROJECT / "docs/v2_round5_cvm_final_acceptance.json"
    cvm = json.loads(cvm_pointer.read_text(encoding="utf-8"))["family_manifest"]
    family_manifest = Path(cvm["path"])
    if sha256_file(family_manifest) != cvm["sha256"]:
        raise ValueError("CVM family identity differs from accepted pointer")
    return root, manifest, family_manifest, accepted


def native_fundamentals(frame: pl.DataFrame) -> pl.DataFrame:
    """Preserve signed physical values and receipt ages before fit-only scaling."""
    earnings = pl.col("earnings_yield_ttm")
    return frame.with_columns(
        pl.when(pl.col("book_to_market") > 0)
        .then(pl.col("book_to_market").log())
        .otherwise(None)
        .alias("book_to_market"),
        pl.when(earnings.is_finite())
        .then((earnings < 0).cast(pl.Float64))
        .otherwise(None)
        .alias("earnings_negative_flag"),
        pl.col("earnings_yield_ttm_age_sessions").alias(
            "earnings_negative_flag_age_sessions"
        ),
    ).select(
        "date", "isin", *NATIVE_FIELDS, *(f"{n}_age_sessions" for n in NATIVE_FIELDS)
    )


def corroborate_u2(
    terms, provider, identity, filings, dates, isins, distribution_changed
):
    """Retrospective evidence only: never a decision-time feature entitlement."""
    candidates = terms.filter(pl.col("evidence").str.starts_with("U2:"))
    reports = unit_action_evidence(
        candidates.select("effective_date", "isin").to_dicts(),
        provider,
        identity,
        filings,
        dates,
        isins,
        distribution_changed,
    )
    originals = {
        (str(r["effective_date"]), r["isin"]): r
        for r in candidates.iter_rows(named=True)
    }
    for report in reports:
        row = originals[report["date"], report["isin"]]
        report.update(old_q=row["shares_per_prior_share"], old_evidence=row["evidence"])
    return reports


def unit_action_evidence(
    events, provider, identity, filings, dates, isins, distribution_changed
):
    """Lookup unit-action evidence independently of prices and inferred factors."""
    sessions = np.asarray(dates, dtype="datetime64[D]")
    names = {n: i for i, n in enumerate(isins)}
    actions = defaultdict(list)
    for row in provider.iter_rows(named=True):
        factor = row["split_factor"]
        if factor is not None and np.isfinite(factor) and factor > 0 and factor != 1:
            actions[row["isin"]].append(row)
    issuer_events = defaultdict(list)
    for row in filings:
        text = normalized(row["kind"] + " " + row["subject"])
        if any(word in text for word in UNIT_WORDS):
            issuer_events[row["cvm_code"].zfill(6)].append(row)
    identities = defaultdict(list)
    for row in (
        identity.select("date", "isin", "cvm_code", "identity_known_date")
        .sort("date")
        .iter_rows(named=True)
    ):
        if row["identity_known_date"] > row["date"]:
            raise ValueError("issuer bridge uses a future identity")
        identities[row["isin"]].append(
            (np.datetime64(row["date"]), row["cvm_code"].zfill(6))
        )
    reports = []
    for row in events:
        event = np.datetime64(row["effective_date"], "D")
        i = int(np.searchsorted(sessions, event))
        if i == len(sessions) or sessions[i] != event or row["isin"] not in names:
            continue
        n = names[row["isin"]]
        reasons = []
        changes = np.flatnonzero(distribution_changed[max(0, i - 2) : i + 3, n]) + max(
            0, i - 2
        )
        for j in changes:
            reasons.append(
                {
                    "kind": "dismes",
                    "date": str(sessions[j]),
                    "offset_sessions": int(j - i),
                }
            )
        for other in actions[row["isin"]]:
            offset = int(np.searchsorted(sessions, np.datetime64(other["ex_date"]))) - i
            if abs(offset) <= 5:
                reasons.append(
                    {
                        "kind": "provider_unit_action",
                        "date": str(other["ex_date"]),
                        "offset_sessions": offset,
                        "source": other["source"],
                        "factor": other["split_factor"],
                    }
                )
        history = identities[row["isin"]]
        prior = [issuer for day, issuer in history if day <= event]
        issuer = prior[-1] if prior else None
        for filing in issuer_events[issuer]:
            offset = (
                int(np.searchsorted(sessions, np.datetime64(filing["receipt"].date())))
                - i
            )
            if abs(offset) <= 30:
                reasons.append(
                    {
                        "kind": "ipe_unit_subject",
                        "date": filing["receipt"].isoformat(),
                        "offset_sessions": offset,
                        "id": filing["id"],
                        "category": filing["kind"],
                        "subject": filing["subject"],
                        "source": filing["source"],
                    }
                )
        reports.append(
            {
                "isin": row["isin"],
                "date": str(event),
                "date_index": i,
                "name_index": n,
                "issuer": issuer,
                "classification": "corroborated_U2"
                if reasons
                else "large_move_no_action",
                "corroboration": reasons,
            }
        )
    return reports


def continued_quotes(manifest, dates, isins, active, output):
    """Scan only special cash records; existing ISIN and prior eligibility bind."""
    parser_path = PROJECT / "collector/scripts/parse_b3_cotahist.py"
    spec = importlib.util.spec_from_file_location("parse_b3_cotahist", parser_path)
    parser = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = parser
    spec.loader.exec_module(parser)
    day_index = {d: i for i, d in enumerate(dates)}
    name_index = {n: i for i, n in enumerate(isins)}
    first_active = {
        n: int(np.flatnonzero(active[:, i])[0])
        for n, i in name_index.items()
        if active[:, i].any()
    }
    retained, rejected, sources = [], defaultdict(int), []
    for source in manifest["metadata"]["cotahist_provenance"]["raw_archives"]:
        path = Path(source["path"])
        year = int(path.stem[-4:])
        if year > DEVELOPMENT_END.year:
            continue
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"immutable COTAHIST source changed: {path}")
        sources.append(source)
        count = 0
        with zipfile.ZipFile(path) as archive:
            with archive.open(parser.choose_txt_member(archive, year)) as handle:
                for line in handle:
                    if line[:2] != b"01" or line[10:12] not in (b"06", b"07", b"08"):
                        continue
                    row = parser.parse_quote_line(line)
                    if row is None:
                        continue
                    d = day_index.get(row["trade_date"])
                    first = first_active.get(row["isin"])
                    if d is None or first is None or d <= first:
                        rejected["outside_axis_or_no_prior_eligibility"] += 1
                        continue
                    retained.append(row)
                    count += 1
        print(
            json.dumps(
                {"phase": "continuation_archive", "year": year, "retained": count}
            ),
            flush=True,
        )
    frame = parser.rows_to_frame(retained)
    if frame.height:
        frame, removed = parser.detect_block_variants(frame.to_dicts())
        frame, duplicates = parser.collapse_security_days(parser.rows_to_frame(frame))
        rejected["block_variants"] += removed
        rejected["exact_duplicates"] += duplicates
        # Positive, coherent quotes only; zero-trade rows remain explicit in audit.
        coherent = (
            (pl.col("close_brl") > 0)
            & (pl.col("open_brl") > 0)
            & (pl.col("low_brl") > 0)
            & (
                pl.col("high_brl")
                >= pl.max_horizontal("open_brl", "close_brl", "low_brl")
            )
            & (pl.col("low_brl") <= pl.min_horizontal("open_brl", "close_brl"))
        )
        rejected["invalid_ohlc"] += frame.filter(~coherent).height
        frame = frame.filter(coherent)
    frame.write_parquet(output / "continued_quotes.parquet")
    summary = (
        frame.group_by("isin", "ticker", "bdi_code")
        .agg(
            pl.col("trade_date").min().alias("first_date"),
            pl.col("trade_date").max().alias("last_date"),
            pl.len().alias("rows"),
        )
        .sort("isin", "first_date")
        if frame.height
        else pl.DataFrame()
    )
    summary.write_parquet(output / "continued_quote_summary.parquet")
    return {
        "rows": frame.height,
        "names": frame["isin"].n_unique() if frame.height else 0,
        "excluded": dict(rejected),
        "sources": sources,
        "data": bind(output / "continued_quotes.parquet"),
    }


def audit(output: Path):
    root, manifest, family_manifest, accepted = registered_sources()
    output.mkdir(parents=True, exist_ok=False)
    axis = np.load(root / "date_index.npy", allow_pickle=False)
    rows = np.arange(len(axis))
    samples = rows[
        (axis <= np.datetime64(PRETRAIN_END)) | (axis >= np.datetime64(FINETUNE_START))
    ]
    store, access = open_store_for_samples(
        root, samples, purpose="training", history_lookbacks=60, history_end_offsets=0
    )
    try:
        dates, isins = store.dates.astype(object).tolist(), store.isins
        active = store.read("active", rows)

        def table(name):
            return pl.read_parquet(root / manifest["tables"][name]["path"])

        family = json.loads(family_manifest.read_text(encoding="utf-8"))
        cvm_root = Path(family["source_root"])
        identity = pl.read_parquet(family_manifest.parent / "identity.parquet")
        filings = rad_rows(cvm_root)
        u2 = corroborate_u2(
            table("corporate_actions_verified_terms"),
            table("corporate_actions_provider_observations"),
            identity,
            filings,
            store.dates,
            isins,
            store.read("distribution_change_mask", rows),
        )
        write_json_atomic(output / "u2_events.json", u2)
        prints = continued_quotes(manifest, dates, isins, active, output)
        source = pl.read_parquet(family_manifest.parent / "fundamentals.parquet")
        native = native_fundamentals(source)
        native.write_parquet(output / "fundamentals_native.parquet")
        values, valid, ages = align_family(native, dates, isins, NATIVE_FIELDS)
        original_fields = manifest["feature_names"]["sidecar_fundamentals"]
        original_mask = store.read("sidecar_fundamentals_valid", rows)
        support = []
        for f, name in enumerate(NATIVE_FIELDS):
            old_name = (
                "earnings_yield_ttm" if name == "earnings_negative_flag" else name
            )
            old = original_mask[..., original_fields.index(old_name)]
            physical = valid[..., f] & active
            sparse = physical.sum(axis=1) < 20
            support.append(
                {
                    "field": name,
                    "valid_active_name_days": int(physical.sum()),
                    "previously_suppressed_active_name_days": int(
                        (physical & ~old).sum()
                    ),
                    "suppressed_on_less_than_20_name_dates": int(
                        (physical & ~old & sparse[:, None]).sum()
                    ),
                }
            )
        write_json_atomic(output / "native_support.json", support)
        report = {
            "schema": "BRAZIL_RV_ROUND7_SOURCE_AUDIT_V1",
            "base_store": accepted["store"],
            "cvm_family_manifest": bind(family_manifest),
            "access": access.payload(),
            "u2_candidates": len(u2),
            "u2_reclassified": sum(
                r["classification"] == "large_move_no_action" for r in u2
            ),
            "u2_events": bind(output / "u2_events.json"),
            "continued_prints": prints,
            "native_family": bind(output / "fundamentals_native.parquet"),
            "native_support": support,
            "peak_rss_bytes": peak_rss_bytes(),
            "scores_read": False,
            "status": "source_audit_complete_foreign_clock_and_store_pending",
        }
        if report["peak_rss_bytes"] > 8 * 1024**3:
            raise MemoryError("Round-7 preflight exceeded 8 GiB")
        write_json_atomic(output / "manifest.json", report)
        return report
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.output)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "u2_candidates",
                    "u2_reclassified",
                    "peak_rss_bytes",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
