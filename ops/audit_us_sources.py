"""Independently check bounded original US bars, return endpoints and close clocks."""

import calendar
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
import json
import math
from pathlib import Path
import re
import time as timer
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import bound_json, source_families

PROJECT = Path(__file__).resolve().parents[1]
NY = ZoneInfo("America/New_York")
SP = ZoneInfo("America/Sao_Paulo")
UTC = timezone.utc


def binding(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def array_field(text, key):
    # Do not decode the response's present-day market-price metadata. The bounded
    # timestamp axis is checked before decoding its historical indicator arrays.
    match = re.search(r'"' + key + r'"\s*:', text)
    return json.JSONDecoder().raw_decode(text[match.end() :].lstrip())[0]


def close_time(day):
    fourth_thursday = [
        week[calendar.THURSDAY]
        for week in calendar.monthcalendar(day.year, 11)
        if week[calendar.THURSDAY]
    ][3]
    thanksgiving_friday = date(day.year, 11, fourth_thursday) + timedelta(days=1)
    early = day == thanksgiving_friday or (
        (day.month, day.day) in ((7, 3), (12, 24)) and day.weekday() <= 3
    )
    return datetime.combine(day, time(13 if early else 16), NY).astimezone(UTC)


def main():
    started = timer.perf_counter()
    pointer = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text()
    )
    output = Path(pointer["root"]) / "us_source_audit.json"
    if output.exists():
        raise FileExistsError(output)
    accepted = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())
    store = Path(accepted["store"]["root"])
    manifest = bound_json(
        {
            "path": str(store / "manifest.json"),
            "sha256": accepted["store"]["manifest_sha256"],
        }
    )
    parent = manifest["metadata"]["data_repair"]["parent"]
    families = source_families(
        bound_json(
            {
                "path": str(Path(parent["root"]) / "manifest.json"),
                "sha256": parent["manifest_sha256"],
            }
        )
    )
    family = bound_json(families["cross_market"]["source_manifest"])
    root = Path(family["sources"]["us_returns"]["path"]).parent
    source_manifest = root / "manifest.json"
    source = json.loads(source_manifest.read_text())
    assert (
        source["outputs"]["us_returns.parquet"]["sha256"]
        == family["sources"]["us_returns"]["sha256"]
    )
    acquisition = bound_json(source["acquisition"])
    documentation = bound_json(source["source_documentation"])
    frames = {}
    for name in ("us_daily.parquet", "us_returns.parquet"):
        record = source["outputs"][name]
        assert sha256_file(Path(record["path"])) == record["sha256"]
        frames[name] = pl.read_parquet(record["path"])
        assert frames[name]["reference_date"].max() <= date(2024, 12, 31)
    days = np.load(store / "date_index.npy").astype(object)
    decisions = np.array(
        [datetime.combine(d, time(15, 45), SP).timestamp() for d in days]
    )
    normalized = {
        (r["symbol"], r["reference_date"]): r
        for r in frames["us_daily.parquet"].to_dicts()
    }
    original = {}
    receipts = []
    counts = Counter()
    early_examples = []
    for receipt in acquisition["records"]:
        if receipt.get("family") != "us":
            continue
        path = Path(receipt["path"])
        assert sha256_file(path) == receipt["sha256"]
        symbol = receipt["key"]
        text = path.read_text(encoding="utf-8")
        assert re.search(r'"symbol"\s*:\s*"' + symbol + '"', text)
        assert re.search(r'"currency"\s*:\s*"USD"', text)
        dates = [
            datetime.fromtimestamp(t, UTC).astimezone(NY).date()
            for t in array_field(text, "timestamp")
        ]
        assert all(date(2010, 1, 1) <= d <= date(2024, 12, 31) for d in dates)
        indicators = array_field(text, "indicators")
        quote = indicators["quote"][0]
        adjusted = indicators["adjclose"][0]["adjclose"]
        for i, day in enumerate(dates):
            key = (symbol, day)
            assert key not in original
            row = normalized[key]
            values = {
                field: quote[field][i]
                for field in ("open", "high", "low", "close", "volume")
            }
            values["adjusted_close"] = adjusted[i]
            assert all(row[field] == value for field, value in values.items())
            available = (
                None
                if symbol == "SUZ" and day < date(2018, 12, 10)
                else close_time(day)
            )
            assert row["available_at"] == available
            assert row["source_file"] == str(path)
            original[key] = {**values, "available_at": available}
            counts["original_rows"] += 1
            counts["unverified_suz_rows"] += available is None
            if available is not None:
                first = int(
                    np.searchsorted(decisions, available.timestamp(), side="left")
                )
                if first < len(days) and days[first] == day:
                    counts["same_date_first_decision_rows"] += 1
                    if symbol == "EWZ":
                        early_examples.append(
                            {
                                "reference_date": str(day),
                                "available_utc": available.isoformat(),
                                "b3_local_decision": datetime.combine(
                                    day, time(15, 45), SP
                                ).isoformat(),
                            }
                        )
        receipts.append(receipt)
    assert len(original) == len(normalized)
    us_dates = sorted(day for symbol, day in original if symbol == "EWZ")
    us_index = {day: i for i, day in enumerate(us_dates)}
    expected = {}
    for symbol in sorted({symbol for symbol, _ in original}):
        previous = None
        for day in sorted(day for s, day in original if s == symbol):
            now = original[symbol, day]
            before = None if previous is None else original[symbol, previous]
            adjacent = (
                previous in us_index
                and day in us_index
                and us_index[day] == us_index[previous] + 1
            )
            valid = before is not None and all(
                isinstance(x, (int, float)) and math.isfinite(x) and x > 0
                for row in (before, now)
                for x in (row["adjusted_close"], row["volume"])
            )
            if adjacent and valid:
                expected["us_" + symbol, day] = (
                    previous,
                    now["available_at"],
                    math.log(now["adjusted_close"] / before["adjusted_close"]),
                )
            previous = day
    for row in frames["us_returns.parquet"].to_dicts():
        prior, available, ret = expected.pop((row["series"], row["reference_date"]))
        assert (row["previous_date"], row["available_at"], row["log_return"]) == (
            prior,
            available,
            ret,
        )
        counts["returns"] += 1
        counts["untimed_returns"] += available is None
    assert not expected
    report = {
        "source_manifest": binding(source_manifest),
        "accepted_family": families["cross_market"]["source_manifest"],
        "acquisition": source["acquisition"],
        "documentation": source["source_documentation"],
        "documentation_record_count": len(documentation),
        "original_receipts": receipts,
        "reproducer": binding(Path(__file__)),
        "counts": dict(counts),
        "same_date_ewz_examples": early_examples,
        "mismatches": 0,
        "metadata_market_payload_decoded": False,
        "heldout_historical_payload_decoded": False,
        "changed_store_or_source_arrays": 0,
        "limitations": [
            "Single preserved Yahoo vintage; historical correction/rounding revision share unknown. Later common scale factors cancel in returns but are not proof of vintage fidelity.",
            "Prices are adjusted vendor levels, not contemporaneous unadjusted cash prices or evidence of ADR conversion ratios. Premium fields remain unavailable.",
            "SUZ pre-NYSE segment remains preserved but untimed; this audit does not invent OTC close times.",
            "NYSE early-close source conventions are independently calendared; actual first eligible decision is computed, not a blanket one-session lag.",
            "Only the US original-source-to-normalized-return boundary; other cross-market feeds, financial vintages/denominators and full identity/wealth/label audit remain.",
        ],
        "elapsed_seconds": timer.perf_counter() - started,
    }
    write_json_atomic(output, report)
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "sources": len(receipts),
                "mismatches": 0,
                "seconds": report["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
