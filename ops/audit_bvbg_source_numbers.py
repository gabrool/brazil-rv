"""Independent original IN/PR arithmetic; resumable by bound publication pair.

This does not call the production XML selector, normalizer or feature reducers.
Only the already admitted 2019--2024 publications are decoded. Earlier original
versions retained for three documented exceptional reports remain explicit.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import date, datetime
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
from time import perf_counter
import traceback
import zipfile
from zoneinfo import ZoneInfo

os.environ.setdefault("POLARS_MAX_THREADS", "1")

from lxml import etree
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]
CASH = {
    "quantity": "FinInstrmAttrbts/FinInstrmQty",
    "trades": "TradDtls/TradQty",
    "volume_brl": "FinInstrmAttrbts/NtlFinVol",
    "regular_quantity": "FinInstrmAttrbts/RglrTraddCtrcts",
    "nonregular_quantity": "FinInstrmAttrbts/NonRglrTraddCtrcts",
}


def at(element, path):
    return element.find("/".join("{*}" + x for x in path.split("/")))


def value(element, path):
    found = at(element, path)
    return found.text if found is not None else None


def records(handle, tag):
    for _, element in etree.iterparse(
        handle,
        events=("end",),
        tag="{*}" + tag,
        resolve_entities=False,
        no_network=True,
    ):
        yield element
        element.clear()
        # A Document/AppHdr surrounds each record, so release those too.
        node = element
        while node is not None:
            while node.getprevious() is not None:
                del node.getparent()[0]
            node = node.getparent()


@contextmanager
def original_xml(source, expected, available):
    raw = Path(source["path"]).read_bytes()
    assert sha256(raw).hexdigest() == source["sha256"], source
    with zipfile.ZipFile(io.BytesIO(raw)) as outer:
        members = [x for x in outer.infolist() if not x.is_dir()]
        assert len(members) == 1
        nested = outer.read(members[0])
    with zipfile.ZipFile(io.BytesIO(nested)) as archive:
        stamps = []
        for name in archive.namelist():
            if not name.endswith(".xml"):
                continue
            with archive.open(name) as stream:
                header = stream.read(8192)
            stamp = re.search(rb"<(?:\w+:)?CreDtAndTm>([^<]+)</", header)
            assert stamp is not None, name
            clock = datetime.fromisoformat(stamp[1].decode())
            if clock.tzinfo is None:
                clock = clock.replace(tzinfo=ZoneInfo("America/Sao_Paulo"))
            stamps.append((clock, name))
        cutoff = datetime.fromisoformat(available + "T15:45:00").replace(
            tzinfo=ZoneInfo("America/Sao_Paulo")
        )
        selected = max(x for x in stamps if x[0] <= cutoff)
        assert selected[1] == expected["selected_member"]
        assert selected[0].isoformat() == expected["selected_creation_time"]
        assert (
            sum(x[0] > cutoff for x in stamps)
            == expected["versions_after_decision_excluded"]
        )
        assert sorted(
            (x["creation_time"], x["member"]) for x in expected["versions"]
        ) == sorted((x.isoformat(), n) for x, n in stamps)
        with archive.open(selected[1]) as stream:
            yield (
                stream,
                {
                    **source,
                    "selected_member": selected[1],
                    "creation_time": selected[0].isoformat(),
                    "versions": len(stamps),
                    "after_decision": sum(x[0] > cutoff for x in stamps),
                    "xml_bytes": archive.getinfo(selected[1]).file_size,
                },
            )


def compare_rows(generated, source, day, available):
    frame = pl.read_parquet(source["path"])
    assert binding(Path(source["path"]))["sha256"] == source["sha256"]
    expected = {r["isin"]: r for r in frame.to_dicts()}
    assert len(expected) == frame.height
    assert set(expected) == set(generated), (
        "identity_rows",
        day,
        set(expected) ^ set(generated),
    )
    differences, checks = [], 0
    for isin, row in expected.items():
        actual = {
            "isin": isin,
            "source_trade_date": date.fromisoformat(day),
            "available_date": date.fromisoformat(available),
            **generated[isin],
        }
        for name, v in row.items():
            checks += 1
            if v != actual[name]:
                differences.append(
                    {"isin": isin, "field": name, "expected": v, "actual": actual[name]}
                )
    return {"rows": frame.height, "comparisons": checks, "differences": differences}


def audit_pair(job, identities):
    start = perf_counter()
    day, available = job["day"], job["available"]
    cash, options, units = {}, {}, Counter()
    sources = []
    if "IN" in job:
        with original_xml(job["IN"], job["receipt"]["in"], available) as (
            stream,
            receipt,
        ):
            for e in records(stream, "Instrm"):
                if value(e, "RptParams/ActvtyInd") != "true":
                    continue
                ident = value(e, "FinInstrmId/OthrId/Id")
                equity = at(e, "InstrmInf/EqtyInf")
                option = at(e, "InstrmInf/OptnOnEqtsInf")
                if equity is not None and value(e, "FinInstrmAttrCmon/Mkt") == "10":
                    isin = value(equity, "ISIN")
                    if any(a <= day <= b for a, b in identities.get(isin, [])):
                        assert ident not in cash, (day, ident)
                        cash[ident] = isin
                        units[
                            "cash:"
                            + str(value(equity, "TradgCcy"))
                            + ":price_factor="
                            + str(value(equity, "PricFctr"))
                        ] += 1
                elif option is not None:
                    first, last, expiry = [
                        value(option, x)
                        for x in ("TradgStartDt", "TradgEndDt", "XprtnDt")
                    ]
                    if first and last and expiry and first <= day <= min(last, expiry):
                        assert ident not in options, (day, ident)
                        options[ident] = (
                            value(option, "UndrlygInstrmId/OthrId/Id"),
                            value(option, "OptnTp"),
                            value(option, "TckrSymb"),
                            value(option, "TradgCcy"),
                            value(option, "PricFctr"),
                            value(option, "AllcnRndLot"),
                        )
            sources.append(receipt)
        options = {key: v for key, v in options.items() if v[0] in cash}
        assert len(cash) == job["receipt"]["stock_instruments"]
        assert len(options) == job["receipt"]["mapped_active_series"]
    aggregate = defaultdict(
        lambda: dict(
            listed_series=0,
            reported_series=0,
            oi_observed_series=0,
            call_oi=0,
            put_oi=0,
        )
    )
    for underlying, _, _, currency, factor, lot in options.values():
        aggregate[cash[underlying]]["listed_series"] += 1
        units[f"option:{currency}:price_factor={factor}:round_lot={lot}"] += 1
    cash_rows, seen = {}, set()
    series_reported, series_observed = 0, 0
    with original_xml(job["PR"], job["receipt"]["pr"], available) as (stream, receipt):
        for e in records(stream, "PricRpt"):
            ident = value(e, "FinInstrmId/OthrId/Id")
            ticker = value(e, "SctyId/TckrSymb")
            cash_isin = (
                cash.get(ident) if "IN" in job else job["cash_tickers"].get(ticker)
            )
            if cash_isin is None and ident not in options:
                continue
            assert value(e, "TradDt/Dt") == day
            assert ident not in seen, (day, ident)
            seen.add(ident)
            if ident in options:
                underlying, kind, expected_ticker, *_ = options[ident]
                assert ticker == expected_ticker
                agg = aggregate[cash[underlying]]
                agg["reported_series"] += 1
                series_reported += 1
                oi = value(e, "FinInstrmAttrbts/OpnIntrst")
                if oi is not None:
                    assert re.fullmatch(r"[0-9]+(?:\.0+)?", oi), (day, ticker, oi)
                    assert kind in ("CALL", "PUT", "PUTT")
                    agg["oi_observed_series"] += 1
                    # Exact integer units, independent of production float summation.
                    agg["call_oi" if kind == "CALL" else "put_oi"] += int(
                        oi.split(".")[0]
                    )
                    series_observed += 1
            if cash_isin is not None:
                assert cash_isin not in cash_rows
                row = {}
                for name, path in CASH.items():
                    raw = value(e, path)
                    row[name] = float(raw) if raw is not None else None
                    if name == "volume_brl" and raw is not None:
                        assert at(e, path).get("Ccy") == "BRL"
                cash_rows[cash_isin] = row
        sources.append(receipt)
    for row in aggregate.values():
        row["oi_all_listed_observed"] = (
            row["listed_series"] == row["oi_observed_series"]
        )
    result = {
        "id": job["id"],
        "day": day,
        "available": available,
        "sources": sources,
        "source_receipt": job["source_receipt"],
        "units": dict(units),
        "listed_option_series": len(options),
        "reported_option_series": series_reported,
        "oi_observed_series": series_observed,
        "outputs": {},
        "position_date_disposition": job["receipt"].get(
            "source_position_date",
            "prior B3 session under admitted publication contract",
        ),
    }
    for role, rows in (("oi", aggregate), ("cash", cash_rows)):
        if role in job:
            result["outputs"][role] = compare_rows(rows, job[role], day, available)
            result["outputs"][role]["source"] = job[role]
    result["differences"] = sum(
        len(x["differences"]) for x in result["outputs"].values()
    )
    result["seconds"] = perf_counter() - start
    return result


def work(job, identities, output, code_sha):
    try:
        result = audit_pair(job, identities)
        result["status"] = "passed" if result["differences"] == 0 else "differences"
    except Exception:
        result = {
            "id": job["id"],
            "status": "error",
            "traceback": traceback.format_exc(),
        }
    result["executed_code_sha256"] = code_sha
    write_json_atomic(Path(output) / "days" / (job["id"] + ".json"), result)
    return result


def inventory(run):
    admission = bound_json(run["remaining_auxiliaries"])
    family_path = Path(
        admission["parent_families"]["options"]["source_manifest"]["path"]
    )
    family = bound_json(admission["parent_families"]["options"]["source_manifest"])
    # The family root is an archived producer manifest, not a timestamp search.
    root = family_path.parent.parent
    entries = json.loads((family_path.parent / "source_inventory.json").read_text())
    by_path = {str(Path(x["path"])): x for x in entries}
    raw = json.loads((root / "b3_bvbg_raw/manifest.json").read_text())
    raw_by_name = {
        x["name"]: {
            "path": str(root / "b3_bvbg_raw" / x["filename"]),
            "sha256": x["sha256"],
        }
        for x in raw["files"]
        if "filename" in x
    }
    master_audit = json.loads((root / "b3_identity_axis_audit.json").read_text())
    master = master_audit["master_file"]
    if isinstance(master, str):
        master = {"path": master, "sha256": master_audit["master_sha256"]}
    master_path = Path(master.get("path", master.get("filename", "")))
    assert binding(master_path)["sha256"] == master["sha256"]
    identities = defaultdict(list)
    for row in pl.read_parquet(master_path).to_dicts():
        identities[row["isin"]].append((str(row["first_date"]), str(row["last_date"])))
    jobs = []
    for entry in entries:
        p = Path(entry["path"])
        if not p.name.endswith("_oi.parquet"):
            continue
        stem = p.name.removesuffix("_oi.parquet")
        receipt_path = p.with_name(stem + ".json")
        receipt_binding = by_path.get(str(receipt_path), binding(receipt_path))
        receipt = bound_json(receipt_binding)
        day = receipt["source_trade_date"]
        job = {
            "id": stem,
            "day": day,
            "available": receipt["available_date"],
            "receipt": receipt,
            "oi": entry,
            "source_receipt": receipt_binding,
        }
        cash_path = str(p.with_name(stem + "_cash.parquet"))
        if cash_path in by_path:
            job["cash"] = by_path[cash_path]
        if "raw_files" in receipt:
            job.update(receipt["raw_files"])
        else:
            key = date.fromisoformat(day).strftime("%y%m%d")
            job["IN"] = raw_by_name["IN" + key + ".zip"]
            filename = (
                "PR210104_original_valid_versions.zip"
                if day == "2021-01-04"
                else stem + "_original_versions.zip"
            )
            job["PR"] = {
                "path": str(p.parent / filename),
                "sha256": receipt["retained_pr_sha256"],
            }
            job["original_pr_sha256"] = receipt["original_pr_sha256"]
        jobs.append(job)
    gap = root / "b3_cash_gap_recovery/manifest.json"
    receipt = json.loads(gap.read_text())
    frame = pl.read_parquet(root / "b3_cash_axis/cash.parquet").filter(
        pl.col("source_trade_date") == date(2023, 12, 8)
    )
    tickers = (
        frame.group_by("ticker")
        .agg(pl.col("isin").n_unique().alias("n"), pl.col("isin").first())
        .filter(pl.col("n") == 1)
    )
    jobs.append(
        {
            "id": "2023-12-08_cash_only",
            "day": "2023-12-08",
            "available": "2023-12-11",
            "receipt": {"pr": receipt["publication"]},
            "source_receipt": binding(gap),
            "cash_tickers": dict(tickers.select("ticker", "isin").iter_rows()),
            "PR": raw_by_name["PR231208.zip"],
            "cash": by_path[str(gap.with_name("2023-12-08_cash.parquet"))],
        }
    )
    assert len({j["id"] for j in jobs}) == len(jobs)
    return (
        jobs,
        dict(identities),
        {
            "family": binding(family_path),
            "family_contract": family,
            "inventory": binding(family_path.parent / "source_inventory.json"),
            "master": master,
            "raw_inventory": binding(root / "b3_bvbg_raw/manifest.json"),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    start = perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    output = Path(run["root"]) / "bvbg_source_numbers"
    (output / "days").mkdir(parents=True, exist_ok=True)
    code_sha = binding(Path(__file__))["sha256"]
    (output / ("executed_" + code_sha[:12] + ".py")).write_bytes(
        Path(__file__).read_bytes()
    )
    jobs, identities, sources = inventory(run)
    contract = {
        "jobs": jobs,
        "sources": sources,
        "scope": "Every admitted original IN/PR publication pair; no feature/model rebuild",
        "held_out_read": False,
    }
    contract_path = output / "contract.json"
    if contract_path.exists():
        assert json.loads(contract_path.read_text()) == contract, (
            "Frozen source contract changed"
        )
    else:
        write_json_atomic(contract_path, contract)
    selected = [j for j in jobs if not args.ids or j["id"] in args.ids]
    pending = []
    for job in selected:
        receipt = output / "days" / (job["id"] + ".json")
        if (
            receipt.exists()
            and json.loads(receipt.read_text()).get("status") == "passed"
        ):
            continue
        if receipt.exists():
            prior = sha256(receipt.read_bytes()).hexdigest()[:12]
            receipt.with_name(receipt.stem + "_attempt_" + prior + ".json").write_bytes(
                receipt.read_bytes()
            )
        pending.append(job)
    print(
        json.dumps(
            {"scope_jobs": len(jobs), "pending": len(pending), "workers": args.workers}
        ),
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(work, j, identities, str(output), code_sha) for j in pending
        ]
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["status"] != "passed" or count % 25 == 0 or len(pending) <= 5:
                print(
                    json.dumps(
                        {
                            k: result[k]
                            for k in ("id", "status", "seconds", "traceback")
                            if k in result
                        }
                    ),
                    flush=True,
                )
    receipts = [
        json.loads((output / "days" / (j["id"] + ".json")).read_text())
        for j in jobs
        if (output / "days" / (j["id"] + ".json")).exists()
    ]
    passed = [r for r in receipts if r["status"] == "passed"]
    summary = {
        "status": "complete" if len(passed) == len(jobs) else "incomplete",
        "contract": binding(contract_path),
        "jobs": len(jobs),
        "passed": len(passed),
        "failed": [r["id"] for r in receipts if r["status"] != "passed"],
        "comparisons": sum(
            v["comparisons"] for r in passed for v in r["outputs"].values()
        ),
        "rows": {
            kind: sum(r["outputs"].get(kind, {}).get("rows", 0) for r in passed)
            for kind in ("oi", "cash")
        },
        "listed_option_series": sum(r["listed_option_series"] for r in passed),
        "oi_observed_series": sum(r["oi_observed_series"] for r in passed),
        "worker_seconds": sum(r["seconds"] for r in passed),
        "invocation_seconds": perf_counter() - start,
        "production_data_or_model_changes": False,
    }
    write_json_atomic(output / "manifest.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
