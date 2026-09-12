"""Bounded historical publication-clock audit; no forward capture."""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .artifacts import sha256_file, write_json_atomic
from .round5_derived import bind
from .round7_data import PROJECT


def pdf_creation(value):
    match = re.fullmatch(r"D:(\d{14})(?:([+-])(\d{2})'?([0-9]{2})'?|Z)?", value or "")
    if not match:
        return None
    stamp = datetime.strptime(match[1], "%Y%m%d%H%M%S")
    if match[2]:
        offset = timedelta(hours=int(match[3]), minutes=int(match[4]))
        stamp = stamp.replace(tzinfo=timezone(offset if match[2] == "+" else -offset))
    elif value.endswith("Z"):
        stamp = stamp.replace(tzinfo=UTC)
    else:
        return None  # Do not guess the timezone of a historical timestamp.
    return stamp.astimezone(ZoneInfo("America/Sao_Paulo"))


def audit(output):
    accepted = json.loads(
        (PROJECT / "docs/v2_round6_inputs.json").read_text(encoding="utf-8")
    )
    record = accepted["source_coverage"]["foreign_flow_manifest"]
    source = Path(record["path"])
    if sha256_file(source) != record["sha256"]:
        raise ValueError("foreign-flow source manifest identity changed")
    manifest = json.loads(source.read_text(encoding="utf-8"))
    observations = [r for r in manifest["results"] if r.get("observation")]
    inventory, by_year = [], defaultdict(Counter)
    for row in observations:
        day = date.fromisoformat(row["publication_date"])
        stamp = pdf_creation(row.get("pdf_metadata", {}).get("/CreationDate"))
        status = "missing_or_unzoned_timestamp"
        if stamp is not None:
            status = (
                "regenerated_after_bulletin_date"
                if stamp.date() > day
                else "generation_before_1545"
                if stamp.date() == day and (stamp.hour, stamp.minute) < (15, 45)
                else "generation_after_1545"
                if stamp.date() == day
                else "generation_before_bulletin_date"
            )
        by_year[day.year][status] += 1
        inventory.append(
            {
                "url": row["url"],
                "bulletin_date": str(day),
                "reference_date": row["observation"]["reference_date"],
                "creation_sao_paulo": stamp.isoformat() if stamp else None,
                "status": status,
                "source_sha256": row["sha256"],
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "creation_inventory.json", inventory)
    queries = []
    for year in sorted(by_year):
        query = urllib.parse.urlencode(
            {
                "url": f"arquivos.b3.com.br/bdi/download/bdi/{year}*/BDI_02_*.pdf",
                "output": "json",
                "filter": "statuscode:200",
                "from": str(year),
                "to": "2024",
                "fl": "timestamp,original,digest",
                "collapse": "timestamp:6",
                "limit": "300",
            }
        )
        queries.append("https://web.archive.org/cdx/search/cdx?" + query)

    def retrieve(item):
        index, url = item
        result = {"url": url}
        try:
            with urllib.request.urlopen(
                urllib.request.Request(
                    url, headers={"User-Agent": "BrazilRV historical research"}
                ),
                timeout=20,
            ) as response:
                payload = response.read(5 * 1024**2)
                result["http_status"] = response.status
            path = output / f"wayback_{index}.json"
            path.write_bytes(payload)
            data = json.loads(payload)
            result.update(
                payload=bind(path), captures=max(0, len(data) - 1), status="retrieved"
            )
        except Exception as exc:
            result.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        captures = list(pool.map(retrieve, enumerate(queries)))
    # Captures are evidence to inspect, not an automatic same-day admission.
    report = {
        "schema": "BRAZIL_RV_ROUND7_FOREIGN_CLOCK_AUDIT_V1",
        "source": record,
        "bulletins": len(observations),
        "by_year": {str(y): dict(c) for y, c in by_year.items()},
        "creation_inventory": bind(output / "creation_inventory.json"),
        "wayback_probes": captures,
        "forward_capture": False,
        "methodology_source": "https://www.b3.com.br/pt_br/noticias/dados-do-fluxo-de-investimento-estrangeiro.htm",
        "distinction": "Financial settlement in D+2 does not establish publication before the D+2 decision. A creation timestamp proves generation, not public availability; regeneration is not the original release clock.",
        "current_rule": "Keep printed bulletin date end-of-day, first subsequent B3 decision, unless a capture proves an earlier public availability.",
        "earlier_admission_automatically_authorized": False,
        "consumer_end": "2024-12-30",
    }
    write_json_atomic(output / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.output)
    print(
        json.dumps(
            {
                "by_year": result["by_year"],
                "captures": [
                    {k: v for k, v in r.items() if k != "url"}
                    for r in result["wayback_probes"]
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
