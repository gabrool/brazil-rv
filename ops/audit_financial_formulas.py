"""Independent as-of financial arithmetic against the admitted family parquet."""

from bisect import bisect_right
import argparse
import calendar
from collections import Counter, defaultdict
from datetime import date, datetime, time as daytime, timedelta
import json
import math
from pathlib import Path
import pickle
import statistics
import time
import unicodedata

import numpy as np
import polars as pl

from brazil_rv.v2 import round5_cvm as cvm
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]
FIELDS = (
    "book_to_market",
    "earnings_yield_ttm",
    "gross_profitability",
    "liabilities_to_assets",
    "accruals_to_assets",
    "revenue_growth_yoy",
    "sue",
)


def plain(value):
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", value.lower())
        if not unicodedata.combining(c)
    )


def previous_year(day):
    last = calendar.monthrange(day.year - 1, day.month)[1]
    return day.replace(
        year=day.year - 1, day=last if month_end(day) else min(day.day, last)
    )


def months(start, end):
    return (end.year - start.year) * 12 + end.month - start.month + 1


def month_end(day):
    return day == date(day.year + (day.month == 12), day.month % 12 + 1, 1) - timedelta(
        days=1
    )


def account(document, basis, metric):
    book = document.get("accounts", {}).get(basis, {})
    if metric in book:
        return book[metric]
    if (
        metric == "parent_income"
        and basis == "con"
        and "net_income" in book
        and "minority_income" in book
    ):
        total, minority = book["net_income"], book["minority_income"]
        if (total["start"], total["end"]) == (minority["start"], minority["end"]):
            return dict(total, value=total["value"] - minority["value"])


def pick(books, basis, metric, ending):
    candidates = [
        d
        for d in books
        if d["reference"] == ending and account(d, basis, metric) is not None
    ]
    if not candidates:
        return None
    d = max(candidates, key=lambda d: (d["version"], d["available_index"], d["id"]))
    row = account(d, basis, metric)
    return (row, d) if row["end"] == ending else None


def annual(books, basis, metric, ending):
    current = pick(books, basis, metric, ending)
    if current is None:
        return None
    row, d = current
    start = row["start"]
    if start is None or start.day != 1 or not month_end(ending):
        return None
    n = months(start, ending)
    if n == 12:
        return row["value"], [d]
    if n not in (3, 6, 9):
        return None
    prior = pick(books, basis, metric, previous_year(ending))
    full = pick(books, basis, metric, start - timedelta(days=1))
    if (
        prior is None
        or full is None
        or any(p[0]["start"] != previous_year(start) for p in (prior, full))
    ):
        return None
    return row["value"] + full[0]["value"] - prior[0]["value"], [d, full[1], prior[1]]


def quarters(books, basis, metric):
    accumulated = {}
    for d in sorted(books, key=lambda d: (d["version"], d["available_index"])):
        row = account(d, basis, metric)
        if row is None or row["start"] is None:
            continue
        start, end = row["start"], row["end"]
        n = months(start, end)
        if start.day == 1 and month_end(end) and n in (3, 6, 9, 12):
            accumulated[start, n] = (row, d)
    answer = {}
    for (start, n), (row, d) in accumulated.items():
        prior = accumulated.get((start, n - 3))
        if n == 3 or prior is not None:
            answer[row["end"]] = (
                row["value"] - (prior[0]["value"] if prior else 0),
                [d] + ([prior[1]] if prior else []),
            )
    return answer


def snapshot(books, sector):
    result = {}
    if not books:
        return result, None
    latest = max(books, key=lambda d: (d["reference"], d["version"]))
    basis = next((b for b in ("con", "ind") if latest.get("accounts", {}).get(b)), None)
    if basis is None:
        return result, None
    stock = latest["accounts"][basis]
    end = latest["reference"]
    financial = (
        any(
            w in plain(sector or "")
            for w in (
                "banco",
                "intermediacao financeira",
                "seguros",
                "seguradoras",
                "credito",
            )
        )
        or stock.get("equity", {}).get("source_code") in {"2.05", "2.07", "2.08"}
        or any(
            w in plain(a.get("description", ""))
            for a in stock.values()
            for w in ("intermediacao financeira", "seguradora", "resseguradora")
        )
    )
    assets = stock.get("assets", {}).get("value")
    equity = stock.get("equity", {}).get("value")
    if assets is not None and assets > 0 and equity is not None:
        result["liabilities_to_assets"] = (1 - equity / assets, [latest], end)
    book = equity if basis == "ind" else stock.get("parent_equity", {}).get("value")
    if (
        book is None
        and basis == "con"
        and equity is not None
        and "minority_equity" in stock
    ):
        book = equity - stock["minority_equity"]["value"]
    if book is not None:
        result["book_to_market"] = (book, [latest], end)
    earnings = "parent_income" if basis == "con" else "net_income"
    totals = {
        m: annual(books, basis, m, end)
        for m in ("gross_profit", "revenue", "net_income", "cash_flow", earnings)
    }
    if totals[earnings] is not None:
        result["earnings_yield_ttm"] = (*totals[earnings], end)
    gross = totals["gross_profit"]
    if (
        gross is not None
        and assets is not None
        and assets > 0
        and (
            not financial
            or "intermedia"
            in plain(stock.get("gross_profit", {}).get("description", ""))
        )
    ):
        result["gross_profitability"] = (gross[0] / assets, [latest, *gross[1]], end)
    if not financial:
        current = totals["revenue"]
        prior = annual(books, basis, "revenue", previous_year(end))
        if current is not None and prior is not None and prior[0] != 0:
            result["revenue_growth_yoy"] = (
                (current[0] - prior[0]) / abs(prior[0]),
                current[1] + prior[1],
                end,
            )
        income, cash = totals["net_income"], totals["cash_flow"]
        past = pick(books, basis, "assets", previous_year(end))
        if (
            assets is not None
            and assets > 0
            and past is not None
            and past[0]["value"] > 0
            and income is not None
            and cash is not None
        ):
            result["accruals_to_assets"] = (
                (income[0] - cash[0]) / ((assets + past[0]["value"]) / 2),
                [latest, past[1], *income[1], *cash[1]],
                end,
            )
    q = quarters(books, basis, earnings)
    changes = {
        d: (v[0] - q[previous_year(d)][0], v[1] + q[previous_year(d)][1])
        for d, v in q.items()
        if previous_year(d) in q
    }
    prior = sorted(d for d in changes if d < end)[-8:]
    endpoints = prior + [end]
    if (
        end in changes
        and len(prior) == 8
        and all(
            (b.year - a.year) * 12 + b.month - a.month == 3
            for a, b in zip(endpoints, endpoints[1:])
        )
    ):
        scale = statistics.stdev(changes[d][0] for d in prior)
        if scale > 0:
            result["sue"] = (
                changes[end][0] / scale,
                [doc for d in endpoints for doc in changes[d][1]],
                end,
            )
    return result, financial


def state(books, sector):
    result, financial = snapshot(books, sector)
    missing = set(FIELDS) - set(result)
    if financial:
        missing -= {"revenue_growth_yoy", "accruals_to_assets"}
    for end, version in sorted(
        {(d["reference"], d["version"]) for d in books}, reverse=True
    )[1:]:
        if not missing:
            break
        earlier, old_financial = snapshot(
            [
                d
                for d in books
                if d["reference"] < end
                or (d["reference"] == end and d["version"] <= version)
            ],
            sector,
        )
        if financial is not None and financial != old_financial:
            continue
        for field in missing & set(earlier):
            result[field] = earlier[field]
        missing -= set(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--isin", help="Recheck only this security and its full issuer class set"
    )
    args = parser.parse_args()
    started = time.perf_counter()
    run = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    output = args.output or Path(run["root"]) / "financial_formula_audit"
    output.mkdir(exist_ok=False)
    pointer = json.loads(
        (PROJECT / "docs/v2_data_inputs.json").read_text(encoding="utf8")
    )
    family = bound_json(pointer["financial_family"])
    original = bound_json(family["original_family"])
    propagated = bound_json(run["fca_financial_propagation"])
    identity_source = bound_json(run["fca_identity_admission"])["artifacts"]["identity"]
    for receipt in (identity_source, propagated["artifacts"]["fundamentals"]):
        assert sha256_file(Path(receipt["path"])) == receipt["sha256"]
    identity = pl.read_parquet(identity_source["path"])
    if args.isin:
        issuers_to_check = (
            identity.filter(pl.col("isin") == args.isin)
            .select("cnpj", "cvm_code")
            .unique()
        )
        assert len(issuers_to_check), args.isin
        identity = identity.join(issuers_to_check, on=["cnpj", "cvm_code"], how="inner")
    expected_features = pl.read_parquet(propagated["artifacts"]["fundamentals"]["path"])
    caches = [
        family["extraction"]["cache"],
        propagated["artifacts"]["new_issuer_documents"],
    ]
    documents = []
    for source in caches:
        assert sha256_file(Path(source["path"])) == source["sha256"]
        documents.extend(pickle.loads(Path(source["path"]).read_bytes())[0])
    by_id = {d["id"]: d for d in documents}
    capital = bound_json(run["cvm_capital_admission"])
    for row in capital["sources"]:
        by_id[row["id"]]["shares"] = row["capital"]["shares"]
    store = Path(family["parent"]["root"])
    days = np.load(store / "date_index.npy").astype("datetime64[D]").tolist()
    positions = {day: i for i, day in enumerate(days)}
    columns = {s: i for i, s in enumerate(np.load(store / "isin_index.npy"))}
    cutoff = np.array(
        [datetime.combine(d, daytime(15, 45)) for d in days], dtype="datetime64[us]"
    )
    # Receipt parsing is a separately completed source audit. Here the first
    # usable decision is independently selected from the receipt's upper bound.
    rad = cvm.rad_rows(Path(original["source_root"]))
    receipts = {r["id"]: r for r in rad if r["id"] and r["group"] == "structured"}
    issuers = defaultdict(list)
    for d in documents:
        stamp = receipts.get(d["id"], {}).get("receipt", d["receipt"])
        bound = (
            stamp + timedelta(minutes=1)
            if isinstance(stamp, datetime)
            else datetime.combine(stamp + timedelta(days=1), daytime())
        )
        d["available_index"] = int(np.searchsorted(cutoff, np.datetime64(bound)))
        issuers[d["cnpj"][:8], d["cvm_code"]].append(d)
    observed = np.load(store / "observed.npy", mmap_mode="r")
    prices = np.load(store / "raw_close.npy", mmap_mode="r")
    dist = np.load(store / "distribution_number.npy", mmap_mode="r")
    barrier = np.array(
        np.load(store / "detected_split_mask.npy", mmap_mode="r")
        | np.load(store / "ambiguous_action_mask.npy", mmap_mode="r")
    )
    for col in range(observed.shape[1]):
        rows = np.flatnonzero(observed[:, col] & np.isfinite(dist[:, col]))
        barrier[rows[1:][np.diff(dist[rows, col]) != 0], col] = True
    prefix = np.vstack(
        [np.zeros((1, barrier.shape[1]), dtype=int), np.cumsum(barrier, axis=0)]
    )
    changes = defaultdict(list)
    for event in bound_json(family["capital_changes"]):
        changes[event["cnpj"][:8], event["cvm_code"]].append(
            dict(event, effective=date.fromisoformat(event["effective"]))
        )
    checks = Counter()
    differences = []
    max_errors = defaultdict(float)
    states = 0
    joined = identity.join(
        expected_features, on=["date", "isin"], how="left", validate="1:1"
    )

    def check(field, actual, wanted, info):
        checks[field] += 1
        if actual is None or wanted is None:
            okay = actual is None and wanted is None
        else:
            delta = abs(actual - wanted)
            max_errors[field] = max(max_errors[field], delta)
            okay = math.isclose(actual, wanted, rel_tol=5e-13, abs_tol=1e-10)
        if not okay:
            differences.append(dict(info, field=field, actual=actual, expected=wanted))

    for key, frame in joined.partition_by(["cnpj", "cvm_code"], as_dict=True).items():
        issuer = (key[0][:8], key[1])
        cursor = 0
        known = {}
        calculation = {}
        prior_sector = None
        source = sorted(
            issuers[issuer],
            key=lambda d: (d["available_index"], d["version"], int(d["id"])),
        )
        for (day,), dated in (
            frame.sort("date").partition_by("date", as_dict=True).items()
        ):
            index = positions[day]
            changed = False
            while cursor < len(source) and source[cursor]["available_index"] <= index:
                d = source[cursor]
                cursor += 1
                known[d["kind"], d["reference"], d["version"]] = d
                changed = True
            sector = dated["sector_label"][0]
            books = list(known.values())
            if changed or prior_sector != sector or not calculation:
                calculation = state(books, sector)
                states += 1
            prior_sector = sector
            share_docs = [d for d in books if d.get("shares")]
            capital = max(
                share_docs, key=lambda d: (d["reference"], d["version"]), default=None
            )
            cap = None
            if capital is not None and index > 0:
                start = bisect_right(days, capital["reference"])
                shares = capital["shares"]
                valid = (
                    bool(any(v > 0 for v in shares.values()))
                    and all(math.isfinite(v) and v >= 0 for v in shares.values())
                    and start <= index
                )
                valid &= not any(
                    e["available_index"] <= index
                    and capital["reference"] < e["effective"] <= days[index - 1]
                    for e in changes[issuer]
                )
                value = 0.0
                for cls, q in shares.items():
                    if q <= 0:
                        continue
                    members = dated.filter(pl.col("class") == cls)
                    if len(members) != 1 or (
                        cls == "PN" and members["preferred_class"][0]
                    ):
                        valid = False
                        break
                    col = columns.get(members["isin"][0])
                    if (
                        col is None
                        or not observed[index - 1, col]
                        or prefix[index, col] != prefix[start, col]
                    ):
                        valid = False
                        break
                    price = float(prices[index - 1, col])
                    if not math.isfinite(price) or price <= 0:
                        valid = False
                        break
                    value += q * price
                if valid:
                    cap = value
            for row in dated.iter_rows(named=True):
                info = {"date": str(day), "isin": row["isin"]}
                check(
                    "log_market_cap",
                    row["log_market_cap"],
                    math.log(cap) if cap is not None else None,
                    info,
                )
                for field in FIELDS:
                    item = calculation.get(field)
                    wanted = item[0] if item is not None else None
                    if field in ("book_to_market", "earnings_yield_ttm"):
                        wanted = (
                            wanted / cap
                            if wanted is not None and cap is not None
                            else None
                        )
                    check(field, row[field], wanted, info)
                    if item is not None:
                        assert all(d["available_index"] <= index for d in item[1])
                        latest = max(d["available_index"] for d in item[1])
                        oldest = min(d["available_index"] for d in item[1])
                        if (
                            field in ("book_to_market", "earnings_yield_ttm")
                            and capital is not None
                        ):
                            latest = max(latest, capital["available_index"])
                            oldest = min(oldest, capital["available_index"])
                        check(
                            field + "_age_sessions",
                            row[field + "_age_sessions"],
                            float(index - latest),
                            info,
                        )
                        check(
                            field + "_oldest_dependency_age_sessions",
                            row[field + "_oldest_dependency_age_sessions"],
                            float(index - oldest),
                            info,
                        )
        print(
            json.dumps(
                {
                    "stage": "issuer",
                    "cnpj": key[0],
                    "checks": sum(checks.values()),
                    "differences": len(differences),
                }
            ),
            flush=True,
        )
    write_json_atomic(output / "differences.json", differences)
    report = {
        "schema": "FINANCIAL_FORMULA_AUDIT_V1",
        "requested_security_scope": args.isin,
        "financial_propagation": run["fca_financial_propagation"],
        "identity": identity_source,
        "caches": caches,
        "market_contract": family["parent"],
        "checks": dict(checks),
        "max_absolute_numeric_errors": dict(max_errors),
        "differences": len(differences),
        "recomputed_states": states,
        "rows": len(joined),
        "artifacts": {"differences": binding(output / "differences.json")},
        "reproducer": binding(Path(__file__)),
        "seconds": time.perf_counter() - started,
        "limits": [
            "Independent selected-account arithmetic; original source account extraction is a separate receipt.",
            "Market-cap checks preserve the original market/action-barrier contract; full corrected wealth/history propagation remains.",
            "Own-version choice and coherent fallback are replayed from public receipts, not future replacement filings.",
            "Single archive vintage cannot identify historical revisions. No neural forward or forecast scoring.",
        ],
    }
    write_json_atomic(output / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
