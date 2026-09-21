"""Two narrow meeting windows for renames actually held in added folds."""

from datetime import date
import html
import json
from pathlib import Path
import re
from time import perf_counter

import requests

from brazil_rv.preprocessing.cvm_rad_events import (
    RAD_LIST_ENDPOINT,
    _captcha_setting,
    _rad_payload,
)
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding

PROJECT = Path(__file__).resolve().parents[1]
run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
root = Path(run["scaling_expanded_evaluation_plan"]["path"]).parent
out = root / "rename_meetings"
page = out / "public_query_page.html"
assert _captcha_setting(page.read_bytes()) == "N"
assert "IPE_1_-1_-1" in html.unescape(page.read_text(encoding="utf8"))
(out / "executed.py").write_bytes(Path(__file__).read_bytes())

for year, start, stop, issuer in (
    (2021, date(2021, 6, 7), date(2021, 6, 8), "DEXXOS"),
    (2023, date(2023, 2, 8), date(2023, 2, 9), "WIZ"),
):
    tick = perf_counter()
    payload = _rad_payload(start, stop, ("IPE_1_-1_-1",))
    plan = out / f"{year}_request.json"
    raw = out / f"{year}_response.json"
    assert not raw.exists(), "Reuse completed response"
    write_json_atomic(
        plan,
        dict(
            endpoint=RAD_LIST_ENDPOINT,
            payload=payload,
            selected_issuer=issuer,
            page=binding(page),
        ),
    )
    response = requests.post(RAD_LIST_ENDPOINT, json=payload, timeout=35)
    raw.write_bytes(response.content)
    response.raise_for_status()
    rows = []
    for line in response.json()["d"]["dados"].split("$&&*"):
        fields = line.split("$&")
        if len(fields) < 10:
            continue

        def clean(s):
            return html.unescape(re.sub("<[^>]+>", "", s)).strip()

        company = clean(fields[1])
        if issuer not in company.upper() and not (
            year == 2021 and "GPC" in company.upper()
        ):
            continue
        protocol = re.search(r"NumeroProtocoloEntrega=(\d+)", line)
        rows.append(
            dict(
                year=year,
                cvm=clean(fields[0]),
                issuer=company,
                category=clean(fields[2]),
                subject=clean(fields[4]),
                reference=clean(fields[5]),
                receipt=clean(fields[6]),
                protocol=protocol[1] if protocol else None,
                raw=line,
            )
        )
    destination = root / f"event_sources_{year}"
    destination.mkdir(exist_ok=True)
    write_json_atomic(
        destination / "meeting_index.json",
        dict(
            rows=rows,
            request=binding(plan),
            source_receipts=[binding(raw)],
            seconds=perf_counter() - tick,
            scope="Two-day assembly index, selecting only the actually exposed issuer. No unrelated originals or full-year/source census.",
        ),
    )
    print(
        json.dumps(
            dict(
                year=year,
                selected=[{k: v for k, v in x.items() if k != "raw"} for x in rows],
                seconds=perf_counter() - tick,
            )
        ),
        flush=True,
    )
