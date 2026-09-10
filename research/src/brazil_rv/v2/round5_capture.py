"""Small forward-only B3 index and exact-contract Asian minute snapshots."""

from __future__ import annotations

import argparse
import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .artifacts import sha256_file, write_json_atomic
from .round5_market import _get

INDEX_BASE = "https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall/"
INDEX_METHODS = ("GetPortfolioDay", "GetTheoricalPortfolio", "GetQuartelyPreview")
PRODUCTS = {"I": "dce_iron", "RB": "shfe_rb", "HC": "shfe_hc", "SP": "shfe_sp"}
SHANGHAI = ZoneInfo("Asia/Shanghai")


def index_url(method: str, index: str, page: int = 1) -> str:
    payload = {
        "index": index,
        "language": "pt-br",
        "pageNumber": page,
        "pageSize": 1000,
    }
    encoded = base64.b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode()
    return INDEX_BASE + method + "/" + encoded


def capture_index(root: Path, index: str) -> dict:
    records, views = [], []
    config = _get(
        root / index / "configuration.json", index_url("GetConfigurations", index)
    )
    records.append(config)
    configuration = json.loads(Path(config["path"]).read_text("utf8"))
    # The preview endpoint can return an OLD preview when has=0. Preserve both
    # responses; never infer an active announcement from nonempty data alone.
    preview_active = any(
        item.get("has") == 1 for item in configuration.get("preview", [])
    )
    for method in INDEX_METHODS:
        page, total, count, codes = 1, None, 0, set()
        header = None
        while True:
            source = _get(
                root / index / f"{method}_{page}.json", index_url(method, index, page)
            )
            records.append(source)
            payload = json.loads(Path(source["path"]).read_text("utf8"))
            rows = payload.get("results") or []
            pagination = payload.get("page") or {}
            expected = int(pagination.get("totalRecords", 0))
            if total is None:
                total, header = expected, payload.get("header")
            elif expected != total or payload.get("header") != header:
                raise ValueError("index snapshot changed during pagination")
            for row in rows:
                if row["cod"] in codes:
                    raise ValueError("index pagination duplicated a constituent")
                codes.add(row["cod"])
            count += len(rows)
            if page >= int(pagination.get("totalPages", 1)):
                break
            page += 1
        if count != total or (count == 0 and method != "GetQuartelyPreview"):
            raise ValueError("incomplete current index snapshot")
        views.append({"method": method, "rows": count, "pages": page, "header": header})
    return {
        "family": "index",
        "index": index,
        "status": "captured",
        "preview_active_at_capture": preview_active,
        "views": views,
        "sources": records,
    }


def listed_contracts(
    page: str, prefix: str, captured_at: datetime
) -> tuple[list[str], list[str]]:
    match = re.search(r"futures_app\.inner_futures_per_month\s*=\s*([^;]+);", page)
    if match is None:
        raise ValueError("vendor page has no dated-contract roster")
    codes = json.loads(match.group(1))["data"]
    month = int(captured_at.astimezone(SHANGHAI).strftime("%y%m"))
    current = sorted(
        {
            code
            for code in codes
            if re.fullmatch(prefix + r"\d{4}", code) and int(code[-4:]) >= month
        }
    )
    if not current:
        raise ValueError("vendor roster has no current exact contracts")
    return current, sorted(set(codes) - set(current))


def minute_rows(payload: str) -> list[dict]:
    start, end = payload.find("("), payload.rfind(")")
    if start < 0 or end <= start:
        raise ValueError("minute response is not the documented JSONP format")
    rows = json.loads(payload[start + 1 : end])
    if rows is None:
        return []
    stamps = [datetime.fromisoformat(row["d"]) for row in rows]
    if stamps != sorted(set(stamps)):
        raise ValueError("minute source timestamps are duplicate or unordered")
    return rows


def capture_asia(root: Path, prefix: str, captured_at: datetime) -> dict:
    folder = root / PRODUCTS[prefix]
    page_source = _get(
        folder / "contract_roster.html",
        f"https://finance.sina.com.cn/futures/quotes/{prefix}0.shtml",
    )
    page = Path(page_source["path"]).read_bytes().decode("gb18030", errors="replace")
    contracts, skipped = listed_contracts(page, prefix, captured_at)
    records, summaries = [page_source], []
    for contract in contracts:
        url = (
            "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20_brazil_rv=/"
            f"InnerFuturesNewService.getFewMinLine?symbol={contract}&type=1"
        )
        source = _get(folder / f"{contract}.json", url)
        records.append(source)
        rows = minute_rows(Path(source["path"]).read_text("utf8"))
        summaries.append(
            {
                "contract": contract,
                "rows": len(rows),
                "first_source_wallclock": rows[0]["d"] if rows else None,
                "last_source_wallclock": rows[-1]["d"] if rows else None,
                "status": "raw_minutes_captured" if rows else "vendor_returned_null",
            }
        )
    return {
        "family": "asian_minutes",
        "product": PRODUCTS[prefix],
        "status": "captured",
        "roster_scope": "exact currently unexpired contracts listed on vendor page; not every exchange contract",
        "excluded_continuous_or_expired_symbols": skipped,
        "contracts": summaries,
        "sources": records,
    }


def capture(root: Path, family: str = "all", workers: int = 4) -> dict:
    captured_at = datetime.now(UTC)
    output = root / captured_at.strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    if family in ("all", "index"):
        jobs.extend(("index", index) for index in ("IBOV", "IBXX", "SMLL"))
    if family in ("all", "asia"):
        jobs.extend(("asian_minutes", prefix) for prefix in PRODUCTS)

    def run(job):
        kind, key = job
        try:
            return (
                capture_index(output, key)
                if kind == "index"
                else capture_asia(output, key, captured_at)
            )
        except (OSError, ValueError, KeyError) as error:
            return {"family": kind, "key": key, "status": "failed", "error": str(error)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(run, jobs))
    result = {
        "schema": "BRAZIL_RV_FORWARD_CAPTURE_V1",
        "captured_at_utc": captured_at.isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete"
        if all(row["status"] == "captured" for row in records)
        else "partial_failure",
        "output": str(output),
        "code_sha256": sha256_file(Path(__file__)),
        "development_consumer_allowed": False,
        "first_availability": "UTC per-response retrieved_at is first captured availability; never backdate to source bar/publication date",
        "asian_bar_semantics": "Unmodified vendor one-minute OHLCV/OI payloads and civil timestamps. No filling, continuous stitching or completed-bar guarantee. Historical exchange trade-date/night-session assignment remains a future risk-consumer audit.",
        "retention_evidence": "Initial exact-contract probes returned1023bars spanning>2exchange sessions including nights. One daily capture after23:05Asia/Shanghai overlaps that window; missed runs or changed vendor retention may leave gaps.",
        "records": records,
    }
    write_json_atomic(output / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", choices=["all", "index", "asia"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = capture(args.root, args.family, args.workers)
    print(
        json.dumps(
            {key: result[key] for key in ("status", "output", "completed_at_utc")}
        )
    )
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
