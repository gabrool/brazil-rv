"""Round-5 CVM source recovery and point-in-time issuer observations.

Annual CVM files preserve every filing header, but generally only the latest
version's account/cadastre contents. A header's timestamp must never be applied
to a different version's contents. Earlier contents are recovered from the
version-specific public ENET viewer, with their exact RAD receipt.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import html
import http.cookiejar
import io
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time as clock_time, timedelta
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import polars as pl

END = date(2024, 12, 30)
ENET = "https://www.rad.cvm.gov.br/ENET/"
LEGACY_ANNUAL = Path("C:/quant-data/b3/raw/cvm/snapshot_20260821")
FEATURES_EVENTS = (
    "sessions_since_financial_filing",
    "filing_is_dfp",
    "sessions_since_material_fact",
    "material_fact_count_20",
    "dividend_announcement_age",
    "offering_or_buyback_flag",
    "sessions_until_expected_filing",
)
FEATURES_FUNDAMENTALS = (
    "log_market_cap",
    "book_to_market",
    "earnings_yield_ttm",
    "gross_profitability",
    "liabilities_to_assets",
    "accruals_to_assets",
    "revenue_growth_yoy",
    "sue",
    "statement_age_sessions",
)
ACCOUNTS = {
    "1": "assets",
    "2.03": "equity",
    "2.05": "equity",
    "2.07": "equity",
    "2.08": "equity",
    "2.07.01": "parent_equity",
    "2.03.09": "minority_equity",
    "2.07.02": "minority_equity",
    "2.08.09": "minority_equity",
    "3.01": "revenue",
    "3.03": "gross_profit",
    "3.11": "net_income",
    "3.09": "net_income",
    "3.13": "net_income",
    "3.11.01": "parent_income",
    "3.09.01": "parent_income",
    "3.13.01": "parent_income",
    "3.09.02": "minority_income",
    "3.11.02": "minority_income",
    "3.13.02": "minority_income",
    "6.01": "cash_flow",
}


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def normalized(value: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", value.lower())
        if not unicodedata.combining(c)
    )


def legal_spelling(value: str) -> str:
    """Exact legal spelling apart from accents, punctuation and terminal S.A."""
    value = re.sub(r"[^a-z0-9]", "", normalized(value))
    return value[:-2] if value.endswith("sa") else value


def digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def available_session(receipt: datetime | date, sessions: list[date]) -> int:
    """Minute stamps are upper bounds; date-only receipts are end-of-day.

    A 15:44 receipt enters the 15:45 decision. Historical timezone is Sao Paulo;
    receipt datetimes supplied by RAD are already in that local timezone.
    """
    instant = (
        (receipt + timedelta(minutes=1))
        if isinstance(receipt, datetime)
        else datetime.combine(receipt, clock_time.max)
    )
    index = bisect.bisect_left(sessions, instant.date())
    if (
        index < len(sessions)
        and sessions[index] == instant.date()
        and instant.time() > clock_time(15, 45)
    ):
        index += 1
    return index


def read_csv_member(path: Path, name: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        if name not in archive.namelist():
            return []
        return list(
            csv.DictReader(
                io.StringIO(archive.read(name).decode("latin1")), delimiter=";"
            )
        )


def fetch(url: str, *, opener=None, attempts: int = 3) -> bytes:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                urllib.parse.quote(url, safe="/:?&=+#%"),
                headers={"User-Agent": "Brazil-RV historical research/1.0"},
            )
            with (
                opener.open(request, timeout=60)
                if opener
                else urllib.request.urlopen(request, timeout=60)
            ) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt + 1 == attempts:
                raise
            time.sleep(1 + attempt)
    raise AssertionError("unreachable")


def acquire_annual(root: Path, *, workers: int = 3) -> dict:
    """Reuse immutable local archives; fetch only named development-year files."""
    destination = root / "annual"
    destination.mkdir(parents=True, exist_ok=True)
    requests = [
        (kind, year)
        for kind in ("fca", "dfp", "itr", "ipe", "fre")
        for year in range(2011 if kind == "itr" else 2010, 2025)
    ]

    def acquire(pair):
        kind, year = pair
        filename = f"{kind}_cia_aberta_{year}.zip"
        existing = LEGACY_ANNUAL / filename
        path = existing if existing.exists() else destination / filename
        url = f"https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/{kind.upper()}/DADOS/{filename}"
        try:
            if not path.exists():
                payload = fetch(url)
                if not zipfile.is_zipfile(io.BytesIO(payload)):
                    raise ValueError("HTTP response is not a ZIP archive")
                path.write_bytes(payload)
            with zipfile.ZipFile(path) as archive:
                members = archive.namelist()
            return {
                "kind": kind,
                "year": year,
                "path": str(path.resolve()),
                "url": url,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "members": members,
                "status": "acquired",
                "reused": path == existing,
            }
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            return {
                "kind": kind,
                "year": year,
                "url": url,
                "status": "unavailable",
                "error": str(error),
            }

    results = list(ThreadPoolExecutor(max_workers=workers).map(acquire, requests))
    manifest = {
        "schema": "ROUND5_CVM_ANNUAL_SOURCES_V1",
        "through": str(END),
        "files": results,
        "acquired_bytes": sum(
            x.get("bytes", 0) for x in results if not x.get("reused")
        ),
        "all_contents_vintage": "header_versions_present; detail_versions_must_be_checked_per_document",
    }
    write_json(root / "annual_manifest.json", manifest)
    return manifest


def annual_paths(root: Path, kind: str) -> list[Path]:
    manifest = json.loads((root / "annual_manifest.json").read_text(encoding="utf-8"))
    return [
        Path(row["path"])
        for row in manifest["files"]
        if row["kind"] == kind and row["status"] == "acquired"
    ]


def acquire_rad(root: Path) -> dict:
    from brazil_rv.preprocessing.cvm_rad_events import (
        QUERY_GROUPS,
        RAD_LIST_ENDPOINT,
        RAD_PAGE,
        _captcha_setting,
        _query_periods,
        _rad_payload,
        _request,
    )

    destination = root / "rad"
    destination.mkdir(exist_ok=True)
    page = _request(RAD_PAGE)
    if _captcha_setting(page) != "N":
        raise RuntimeError(
            "RAD requires CAPTCHA; historical source acquisition stopped"
        )
    entries = []
    manifest = {
        "schema": "ROUND5_CVM_RAD_SOURCES_V1",
        "through": str(END),
        "files": entries,
    }
    groups = {
        **QUERY_GROUPS,
        "cadastre": ("EST_1", "EST_2"),
        "offering": ("IPE_78_-1_-1",),
        "proventos": ("IPE_107_-1_-1",),
    }
    (destination / "rad_source_page.html").write_bytes(page)
    manifest["category_evidence_sha256"] = hashlib.sha256(page).hexdigest()
    for group, categories in groups.items():
        for year in range(2010, 2025):
            for period, start, end in _query_periods(year, group):
                end = min(end, END)
                request = _rad_payload(start, end, categories)
                path = destination / f"rad_{year}_{group}_{period}.json"
                if path.exists():
                    payload = path.read_bytes()
                else:
                    payload = _request(RAD_LIST_ENDPOINT, json.dumps(request).encode())
                    path.write_bytes(payload)
                result = json.loads(payload)["d"]
                if (
                    result.get("SolicitarCaptcha") == "S"
                    or result.get("temErro")
                    or result.get("expirouSessao")
                ):
                    write_json(root / "rad_manifest.json", manifest)
                    raise RuntimeError(
                        "RAD unavailable/CAPTCHA: " + str(result.get("msgErro"))
                    )
                entries.append(
                    {
                        "path": str(path),
                        "sha256": sha256(path),
                        "group": group,
                        "year": year,
                        "request": request,
                        "bytes": len(payload),
                    }
                )
                write_json(root / "rad_manifest.json", manifest)
                time.sleep(0.25)
    manifest["status"] = "complete"
    write_json(root / "rad_manifest.json", manifest)
    return manifest


def rad_rows(root: Path) -> list[dict]:
    """Retain legacy timestamped records as events even without an ENET ID."""
    manifest = json.loads((root / "rad_manifest.json").read_text(encoding="utf8"))
    if manifest.get("status") != "complete":
        raise ValueError(
            "RAD event counts require complete registered source-query coverage"
        )
    rows = []
    for source in manifest["files"]:
        data = json.loads(Path(source["path"]).read_text(encoding="utf8"))["d"]["dados"]
        for raw in data.split("$&&*"):
            fields = raw.split("$&")
            if len(fields) < 10:
                continue

            def clean(value):
                return html.unescape(re.sub("<[^>]+>", "", value)).strip()

            timestamp = re.search(
                r"(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})", clean(fields[6])
            )
            if timestamp is None:
                raise ValueError("RAD event lacks its minute receipt")
            receipt = datetime.strptime(" ".join(timestamp.groups()), "%d/%m/%Y %H:%M")
            if receipt.date() > END:
                continue
            ref = re.search(r"\d{2}/\d{2}/\d{4}", clean(fields[5]))
            exact = re.search(r"NumeroSequencialDocumento=(\d+)", raw)
            other = re.search(r"OpenDownloadDocumentos\('([0-9]+)'", raw)
            rows.append(
                {
                    "cvm_code": digits(clean(fields[0])).zfill(6),
                    "kind": clean(fields[2]),
                    "group": source["group"],
                    "reference": datetime.strptime(ref[0], "%d/%m/%Y").date()
                    if ref
                    else None,
                    "receipt": receipt,
                    "version": clean(fields[8]),
                    "id": exact[1] if exact else other[1] if other else None,
                    "subject": clean(fields[4])
                    + " "
                    + clean(fields[11] if len(fields) > 11 else ""),
                    "source": source["path"],
                }
            )
    return rows


def filing_headers(root: Path, kind: str) -> list[dict]:
    output = []
    for path in annual_paths(root, kind):
        for row in read_csv_member(path, path.stem + ".csv"):
            receipt = date.fromisoformat(row["DT_RECEB"])
            if receipt > END:
                continue
            record = {
                "id": digits(row["ID_DOC"]),
                "cnpj": digits(row["CNPJ_CIA"]),
                "cvm_code": row["CD_CVM"].zfill(6),
                "reference": date.fromisoformat(row["DT_REFER"]),
                "version": int(row["VERSAO"]),
                "receipt": receipt,
                "kind": kind.upper(),
                "name": row["DENOM_CIA"],
            }
            output.append(record)
    return output


class Tables(HTMLParser):
    """Read the public viewer's literal table cells without a browser dependency."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row:
                self.rows.append(self.row)
            self.row = None


def table_rows(payload: bytes) -> list[list[str]]:
    parser = Tables()
    parser.feed(payload.decode("utf-8", errors="replace"))
    return parser.rows


def viewer_pages(document: dict, destination: Path) -> dict:
    """Recover exact-version account pages, saving source bytes before parsing.

    The server binds its public viewer Hash to a cookie session. Read the actual
    returned child URL and retain that cookie; do not guess a document-version
    URL or borrow another filing's timestamp. No CAPTCHA bypass is used.
    """
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["status"] == "complete":
            return previous
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    url = (
        ENET
        + "frmGerenciaPaginaFRE.aspx?"
        + urllib.parse.urlencode(
            {"NumeroSequencialDocumento": document["id"], "CodigoTipoInstituicao": 1}
        )
    )
    payload = fetch(url, opener=opener)
    (destination / "viewer.html").write_bytes(payload)
    text = payload.decode("utf-8", errors="replace")
    match = re.search(r"window.frames\[0\].location='([^']+)'", text)
    if match is None:
        raise ValueError("Version-specific viewer omitted its child page")
    child = html.unescape(match[1])
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(child).query)
    if query.get("Versao", [str(document["version"])])[0] != str(document["version"]):
        raise ValueError("Viewer returned another filing version")
    required = {
        "NumeroSequencialDocumento",
        "NumeroSequencialRegistroCvm",
        "CodigoTipoInstituicao",
        "Hash",
        "CodTipoDocumento",
    }
    bound = {k: v[0] for k, v in query.items() if k in required}
    options = [
        html.unescape(value)
        for value in re.findall(r'<option[^>]*value="([^"]+)"', text)
    ]
    urls = [child]
    for option in options:
        if not option.startswith("frmDemonstracaoFinanceiraITR.aspx"):
            continue
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(option).query)
        if q.get("Demonstracao", [""])[0] not in {"2", "3", "4", "99"}:
            continue
        parsed = {k: v[0] for k, v in q.items()}
        parsed.update(bound)
        urls.append(
            "frmDemonstracaoFinanceiraITR.aspx?" + urllib.parse.urlencode(parsed)
        )
    pages = []
    unavailable_pages = []
    seen = set()
    for relative in urls:
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(relative).query)
        key = (q.get("Informacao", [""])[0], q.get("Demonstracao", [""])[0])
        if key in seen:
            continue
        seen.add(key)
        page_url = urllib.parse.urljoin(ENET, relative)
        body = fetch(page_url, opener=opener)
        filename = f"statement_{key[0]}_{key[1]}.html"
        (destination / filename).write_bytes(body)
        rows = table_rows(body)
        if not any(
            len(row) >= 3 and re.fullmatch(r"[1-9](?:\.\d+)*", row[0]) for row in rows
        ):
            unavailable_pages.append(
                {
                    "file": filename,
                    "reason": "No account rows in source page",
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
            )
            continue
        pages.append(
            {
                "file": filename,
                "url": page_url,
                "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body),
                "basis": "con" if key[0] == "2" else "ind",
                "demonstration": key[1],
            }
        )
    if not pages:
        raise ValueError("Viewer returned no account rows in any requested statement")
    manifest = {
        "schema": "ROUND5_CVM_EXACT_VIEWER_V1",
        "status": "complete",
        "document": document,
        "pages": pages,
        "unavailable_pages": unavailable_pages,
        "viewer_sha256": hashlib.sha256(payload).hexdigest(),
    }
    write_json(manifest_path, manifest)
    return manifest


def _account_metric(code: str, description: str) -> str | None:
    metric = ACCOUNTS.get(code)
    if metric is None:
        return None
    description_key = re.sub(r"[^a-z0-9]+", " ", normalized(description)).strip()
    # CVM banks/insurers use several charts: e.g. 2.03 may be deferred
    # income or provisions, and 3.11 may be JCP reversal rather than income.
    # Candidate codes are only a filter; their published descriptions supply
    # the semantic role. Never compare same-number but different accounts.
    if metric == "assets" and description_key not in ("ativo", "ativo total"):
        return
    if metric == "equity" and description_key not in (
        "patrimonio liquido",
        "patrimonio liquido consolidado",
    ):
        return
    if metric == "minority_equity" and "nao controlador" not in description_key:
        return
    if metric == "parent_equity" and not (
        description_key.startswith("patrimonio liquido atribuido")
        and "controlador" in description_key
        and "nao controlador" not in description_key
    ):
        return
    if metric == "net_income" and not (
        description_key.startswith("lucro")
        and ("periodo" in description_key or "exercicio" in description_key)
        and "operacoes" not in description_key
    ):
        return
    if metric == "parent_income" and not (
        "atribuido" in description_key and "empresa controladora" in description_key
    ):
        return
    if metric == "minority_income" and not (
        "atribuido" in description_key and "nao controlador" in description_key
    ):
        return
    return metric


def assign_account(
    document: dict,
    basis: str,
    code: str,
    value: float,
    start: date | None,
    end: date,
    description: str,
    parent_description: str | None = None,
) -> None:
    metric = _account_metric(code, description)
    if metric is None:
        return
    # Some filings retain obsolete child labels under a different chart parent.
    # Use that same statement's parent meaning, independent of source row order.
    if parent_description is not None and metric in (
        "parent_equity",
        "minority_equity",
        "parent_income",
        "minority_income",
    ):
        expected = "equity" if metric.endswith("equity") else "net_income"
        if _account_metric(code.rsplit(".", 1)[0], parent_description) != expected:
            return
    account = {
        "value": value,
        "start": start,
        "end": end,
        "description": description,
        "source_code": code,
    }
    book = document.setdefault("accounts", {}).setdefault(basis, {})
    previous = book.get(metric)
    # ITR DRE often has both a current quarter and a YTD column. Keep YTD;
    # contemporaneously available earlier YTD reports define standalone quarters.
    if previous is None or (
        start is not None
        and previous["start"] is not None
        and start < previous["start"]
    ):
        book[metric] = account
    elif (
        start == previous["start"]
        and end == previous["end"]
        and not np.isclose(value, previous["value"], rtol=1e-12, atol=1e-6)
    ):
        raise ValueError(f"Conflicting account {document['id']}/{basis}/{metric}")


def net_share_counts(paid_in: dict, treasury: dict) -> dict | None:
    """Exact outstanding class counts, never unsigned guesses or missing zeros.

    An explicitly unissued class has zero outstanding even if treasury is
    unreported. Positive paid-in capital requires its actual treasury quantity.
    Negative treasury needs independent source reconciliation, not subtraction.
    """
    result = {}
    for share_class in ("ON", "PN"):
        issued, held = paid_in.get(share_class), treasury.get(share_class)
        try:
            issued = float(issued)
            held = 0.0 if issued == 0 and held in (None, "") else float(held)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(issued) or not np.isfinite(held) or not 0 <= held <= issued:
            return None
        result[share_class] = issued - held
    return result


def load_accounts(root: Path, issuers: set[str] | None = None) -> list[dict]:
    """Read each year's account data once, retaining its own version identity."""
    documents = filing_headers(root, "itr") + filing_headers(root, "dfp")
    if issuers is not None:
        legal_roots = {cnpj[:8] for cnpj in issuers}
        documents = [d for d in documents if d["cnpj"][:8] in legal_roots]
    grouped = defaultdict(list)
    for document in documents:
        grouped[
            (
                document["kind"],
                document["cnpj"],
                str(document["reference"]),
                document["version"],
            )
        ].append(document)
    # A few annual keys have two different IDs with the same version. The
    # account CSV omits ID, so neither ID may borrow those ambiguous contents.
    keyed = {key: group[0] for key, group in grouped.items() if len(group) == 1}
    for kind in ("itr", "dfp"):
        for path in annual_paths(root, kind):
            year = path.stem[-4:]
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    match = re.fullmatch(
                        rf"{kind}_cia_aberta_(BPA|BPP|DRE|DFC_MI|DFC_MD)_(con|ind)_{year}.csv",
                        member,
                    )
                    if match is None:
                        continue
                    statement, basis = match.groups()
                    columns = [
                        "CNPJ_CIA",
                        "DT_REFER",
                        "VERSAO",
                        "MOEDA",
                        "ESCALA_MOEDA",
                        "ORDEM_EXERC",
                        "DT_FIM_EXERC",
                        "CD_CONTA",
                        "DS_CONTA",
                        "VL_CONTA",
                    ]
                    if statement in ("DRE", "DFC_MI", "DFC_MD"):
                        columns.append("DT_INI_EXERC")
                    frame = pl.read_csv(
                        io.BytesIO(archive.read(member).decode("latin1").encode()),
                        separator=";",
                        quote_char=None,
                        infer_schema_length=0,
                        columns=columns,
                    ).filter(
                        pl.col("CD_CONTA").is_in(list(ACCOUNTS))
                        & (pl.col("ORDEM_EXERC") == "ÚLTIMO")
                    )
                    parent_descriptions = {
                        (
                            row["CNPJ_CIA"],
                            row["DT_REFER"],
                            row["VERSAO"],
                            row["CD_CONTA"],
                        ): row["DS_CONTA"]
                        for row in frame.iter_rows(named=True)
                        if row["CD_CONTA"].count(".") == 1
                    }
                    for row in frame.iter_rows(named=True):
                        key = (
                            kind.upper(),
                            digits(row["CNPJ_CIA"]),
                            row["DT_REFER"],
                            int(row["VERSAO"]),
                        )
                        document = keyed.get(key)
                        if (
                            document is None
                            or row["MOEDA"] != "REAL"
                            or row["ESCALA_MOEDA"] not in ("MIL", "UNIDADE")
                        ):
                            continue
                        if row["VL_CONTA"] is None:
                            continue
                        assign_account(
                            document,
                            basis,
                            row["CD_CONTA"],
                            float(row["VL_CONTA"])
                            * (1000 if row["ESCALA_MOEDA"] == "MIL" else 1),
                            date.fromisoformat(row["DT_INI_EXERC"])
                            if row.get("DT_INI_EXERC")
                            else None,
                            date.fromisoformat(row["DT_FIM_EXERC"]),
                            row["DS_CONTA"],
                            parent_descriptions.get(
                                (
                                    row["CNPJ_CIA"],
                                    row["DT_REFER"],
                                    row["VERSAO"],
                                    row["CD_CONTA"].rsplit(".", 1)[0],
                                )
                            ),
                        )
    return documents


def apply_account_unit_dispositions(root: Path, documents: list[dict]) -> dict | None:
    """Apply own-filing currency-unit corrections before any temporal arithmetic.

    The reviewed input rows bind basis, periods and printed values, preventing
    double scaling or silently applying a note to different source contents.
    Receipts, quantities and raw archives are untouched. A note heading alone
    is not a correction: each record carries its actual source reconciliation.
    """
    path = root / "account_unit_dispositions.json"
    if not path.exists():
        return None
    by_id = {str(document["id"]): document for document in documents}
    applied = []
    for record in json.loads(path.read_text(encoding="utf8"))["documents"]:
        identifier = record["document"]["id"]
        if identifier not in by_id:
            continue
        document = by_id[identifier]
        identity = {
            key: str(document[key])
            for key in ("id", "cnpj", "cvm_code", "kind", "reference", "version")
        }
        if identifier in applied or record["document"] != identity:
            raise ValueError("Account unit correction has a different/repeated filing")
        if not record["reason"] or not record["evidence"]:
            raise ValueError("Account unit correction lacks its source reconciliation")
        for source in record["evidence"]:
            if sha256(Path(source["path"])) != source["sha256"]:
                raise ValueError("Account unit correction evidence hash differs")
        accounts = document.get("accounts", {})
        snapshot = json.loads(json.dumps(accounts, default=str))
        if not accounts or snapshot != record["accounts_before"]:
            raise ValueError("Account unit correction differs from the reviewed rows")
        multiplier = float(record["multiplier"])
        if not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("Account currency multiplier must preserve amount signs")
        for metrics in accounts.values():
            for account in metrics.values():
                account["value"] *= multiplier
        applied.append(identifier)
    return {"path": str(path), "sha256": sha256(path), "applied_ids": applied}


def attach_viewer_accounts(document: dict, destination: Path) -> None:
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf8"))
    if (
        manifest["document"]["id"] != document["id"]
        or int(manifest["document"]["version"]) != document["version"]
    ):
        raise ValueError("Recovered viewer identity differs from requested document")
    parsed = {**document, "accounts": {}}
    for page in manifest["pages"]:
        payload = (destination / page["file"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != page["sha256"]:
            raise ValueError("Recovered account page differs from its source manifest")
        text = payload.decode("utf8", errors="replace")
        title = re.search(r"<h2[^>]*>(.*?)</h2>", text, re.S)
        title_text = (
            normalized(html.unescape(re.sub("<[^>]+>", "", title[1]))) if title else ""
        )
        if "reais mil" in title_text:
            scale = 1000
        elif "reais" in title_text:
            scale = 1
        else:
            raise ValueError("Recovered account currency/scale is unavailable")
        rows = table_rows(payload)
        parent_descriptions = {row[0]: row[1] for row in rows if len(row) >= 2}
        header = next((r for r in rows if len(r) > 2 and r[0] == "Conta"), None)
        if header is None:
            raise ValueError("Account table has no dated column header")
        # More than one current-reference column is possible for ITR (quarter/YTD).
        for column in range(2, len(header)):
            dates = [
                datetime.strptime(x, "%d/%m/%Y").date()
                for x in re.findall(r"\d{2}/\d{2}/\d{4}", header[column])
            ]
            if not dates or dates[-1] != document["reference"]:
                continue
            start = dates[0] if len(dates) > 1 else None
            for row in rows:
                if len(row) <= column or row[0] not in ACCOUNTS or not row[column]:
                    continue
                value = float(row[column].replace(".", "").replace(",", ".")) * scale
                assign_account(
                    parsed,
                    page["basis"],
                    row[0],
                    value,
                    start,
                    dates[-1],
                    row[1],
                    parent_descriptions.get(row[0].rsplit(".", 1)[0]),
                )
    if not parsed["accounts"]:
        raise ValueError("Recovered account pages lack the requested reference period")
    document["accounts"] = parsed["accounts"]
    document["recovered_original"] = True


def recovery_inventory(documents: list[dict]) -> dict:
    missing = [d for d in documents if not d.get("accounts")]
    return {
        "eligible_documents": len(documents),
        "missing_original_account_documents": len(missing),
        "missing_first_versions": sum(d["version"] == 1 for d in missing),
        "estimated_html_bytes_at_250kb_per_document": len(missing) * 250_000,
        "estimated_requests_at_5_per_document": len(missing) * 5,
        "missing": [
            {
                k: d[k]
                for k in (
                    "id",
                    "cnpj",
                    "cvm_code",
                    "reference",
                    "version",
                    "receipt",
                    "kind",
                )
            }
            for d in missing
        ],
    }


def recover_originals(root: Path, documents: list[dict], *, workers: int = 3) -> dict:
    missing = [d for d in documents if not d.get("accounts")]
    write_json(root / "recovery_inventory.json", recovery_inventory(documents))
    records = []

    def recover(document):
        destination = root / "originals" / document["id"]
        try:
            result = viewer_pages(document, destination)
            return {
                "id": document["id"],
                "status": "recovered",
                "bytes": sum(p["bytes"] for p in result["pages"]),
            }
        except (OSError, ValueError, KeyError, TypeError) as error:
            write_json(
                destination / "failure.json",
                {"id": document["id"], "error": str(error)},
            )
            try:
                result = recover_original_zip(
                    document, root / "original_zips" / document["id"]
                )
                return {
                    "id": document["id"],
                    "status": "recovered_zip",
                    "bytes": result["bytes"],
                }
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                StopIteration,
                zipfile.BadZipFile,
            ) as zip_error:
                return {
                    "id": document["id"],
                    "status": "unavailable",
                    "reason": str(error),
                    "original_zip_reason": str(zip_error),
                }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(recover, missing):
            records.append(record)
            if len(records) % 50 == 0:
                write_json(
                    root / "recovery_progress.json",
                    {
                        "processed": len(records),
                        "total": len(missing),
                        "records": records,
                    },
                )
    result = {"processed": len(records), "total": len(missing), "records": records}
    write_json(root / "recovery_result.json", result)
    return result


def recover_original_zip(document: dict, destination: Path) -> dict:
    """Failure-only original-package recovery, never a mass ZIP replacement."""
    from .round5_cvm_xml import original_accounts

    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "source.zip"
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        if sha256(path) != manifest["sha256"]:
            raise ValueError("Original financial ZIP differs from its source manifest")
        return manifest
    url = (
        "https://www.rad.cvm.gov.br/ENETCONSULTA/frmDownloadDocumento.aspx?"
        + urllib.parse.urlencode(
            {"CodigoInstituicao": 1, "NumeroSequencialDocumento": document["id"]}
        )
    )
    if not path.exists():
        payload = fetch(url)
        if not zipfile.is_zipfile(io.BytesIO(payload)):
            failed = destination / (
                "unavailable_" + hashlib.sha256(payload).hexdigest()[:16] + ".bin"
            )
            if not failed.exists():
                failed.write_bytes(payload)
            raise ValueError("Original download returned a non-ZIP response")
        path.write_bytes(payload)
    original_accounts(document, path)
    manifest = {
        "schema": "ROUND5_CVM_ORIGINAL_ZIP_V1",
        "document": document,
        "url": url,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    write_json(manifest_path, manifest)
    return manifest


def fca_documents(
    root: Path, sessions: list[date], rad: list[dict] | None = None
) -> list[dict]:
    headers = filing_headers(root, "fca")
    by_id = {d["id"]: d for d in headers}
    for path in annual_paths(root, "fca"):
        year = path.stem[-4:]
        for row in read_csv_member(path, f"fca_cia_aberta_geral_{year}.csv"):
            document = by_id.get(digits(row["ID_Documento"]))
            if document is not None:
                document["sector_label"] = row["Setor_Atividade"]
                document["legal_name"] = row["Nome_Empresarial"]
                document["securities"] = []
                document["metadata_source"] = "annual_csv"
        for row in read_csv_member(path, f"fca_cia_aberta_valor_mobiliario_{year}.csv"):
            document = by_id.get(digits(row["ID_Documento"]))
            if (
                document is None
                or "securities" not in document
                or row["Mercado"] != "Bolsa"
            ):
                continue
            kind = normalized(row["Valor_Mobiliario"])
            if not (
                kind.startswith("acoes ordinarias")
                or kind.startswith("acoes preferenciais")
                or kind == "acoes"
                or kind == "units"
            ):
                continue
            document["securities"].append(
                {
                    "ticker": row["Codigo_Negociacao"].strip().upper(),
                    "class": "SHARES"
                    if kind == "acoes"
                    else "ON"
                    if kind.startswith("acoes ordinarias")
                    else "PN"
                    if kind.startswith("acoes preferenciais")
                    else "UNIT",
                    "preferred_class": row["Sigla_Classe_Acao_Preferencial"],
                    "unit_composition": row["Composicao_BDR_Unit"],
                    "start": date.fromisoformat(row["Data_Inicio_Listagem"])
                    if row["Data_Inicio_Listagem"]
                    else date.min,
                    "end": date.fromisoformat(row["Data_Fim_Listagem"])
                    if row["Data_Fim_Listagem"]
                    else date.max,
                    "segment_start": row["Data_Inicio_Negociacao"],
                    "segment_end": row["Data_Fim_Negociacao"],
                }
            )
    exact = {r["id"]: r for r in rad or [] if r["id"] and r["group"] == "cadastre"}
    parsed_fca_helper_sha = sha256(Path(__file__).with_name("round5_cvm_fca.py"))
    for document in headers:
        original = root / "fca_originals" / document["id"]
        if (original / "manifest.json").exists():
            from .round5_cvm_fca import load_fca

            metadata = load_fca(document, original)
            captured = json.loads(
                (original / "manifest.json").read_text(encoding="utf8")
            )
            document["original_fca_source"] = {
                "document_id": document["id"],
                "manifest_path": str(original / "manifest.json"),
                "manifest_sha256": sha256(original / "manifest.json"),
                "captured_helper_sha256": captured.get("helper_sha256"),
                "parsed_helper_sha256": parsed_fca_helper_sha,
            }
            if metadata is not None:
                document.update(metadata)
                document["sector_label"] = metadata.get("sector")
                document.pop("sector", None)
                for security in document["securities"]:
                    # These are exchange listing/cancellation bounds. Segment
                    # negotiation dates do not restart an existing listing.
                    for key in ("start", "end"):
                        security[key] = date.fromisoformat(security[key])
        receipt = (
            exact[document["id"]]["receipt"]
            if document["id"] in exact
            else document["receipt"]
        )
        document["available_index"] = available_session(receipt, sessions)
        code = document.get("sector_code")
        document["sector_code"] = (
            str(int(code)) if code is not None and str(code).strip() else None
        )
    return headers


def build_identity(
    documents: list[dict],
    observations: pl.DataFrame,
    sessions: list[date],
    isins: list[str],
) -> pl.DataFrame:
    """Join known FCA tickers or exact historical legal names to dated ISINs.

    A missing original FCA version contributes no invented mapping. Sector is
    the versioned FCA classification, not a current B3 sector retrojection.
    Name fallback requires a unique contemporaneous CNPJ and an instrument
    already observed before that FCA receipt. Original generic Ações can map
    only to independently observed ON/PN classes through that exact name route.
    New securities cannot inherit an old company's spelling.
    Explicit sector codes enter at their own receipt. Label-only sectors use
    only unambiguous code/label evidence already received by this decision;
    later evidence never fills or removes an earlier sector classification.
    """
    universe = set(isins)
    events = defaultdict(list)
    for d in documents:
        events[d["available_index"]].append(d)
    prices_by_date = defaultdict(list)
    for row in observations.iter_rows(named=True):
        if row["isin"] in universe:
            prices_by_date[row["trade_date"]].append(row)
    current = {}
    ticker_isin = {}
    known_security = {}
    first_seen = {}
    codes_by_label = defaultdict(dict)

    def observe(row):
        ticker_isin[row["ticker"]] = row["isin"]
        known_security[row["isin"]] = row
        first_seen.setdefault(row["isin"], row["trade_date"])

    for observed_date in sorted(d for d in prices_by_date if d < sessions[0]):
        for row in prices_by_date[observed_date]:
            observe(row)
    output = []
    for index, current_date in enumerate(sessions):
        if index:
            for row in prices_by_date[sessions[index - 1]]:
                observe(row)
        for d in sorted(events[index], key=lambda d: (d["receipt"], d["version"])):
            code, label = d.get("sector_code"), d.get("sector_label")
            if code is not None and label:
                codes_by_label[normalized(label)].setdefault(code, d)
            if "securities" not in d:
                continue
            issuer_key = (d["cnpj"][:8], d["cvm_code"])
            prior = current.get(issuer_key)
            if prior is None or (d["reference"], d["version"]) > (
                prior["reference"],
                prior["version"],
            ):
                current[issuer_key] = d
        mapped = {}
        legal_issuers = defaultdict(set)
        name_isins = defaultdict(list)
        for document in current.values():
            cnpj = document["cnpj"]
            if document.get("legal_name"):
                legal_issuers[legal_spelling(document["legal_name"])].add(cnpj)
        for isin, observed in known_security.items():
            name_isins[legal_spelling(observed.get("issuer_short_name", ""))].append(
                isin
            )
        for document in current.values():
            cnpj = document["cnpj"]
            sector, mapping = document.get("sector_code"), None
            if sector is not None:
                mapping = document
            else:
                candidates = codes_by_label[
                    normalized(document.get("sector_label") or "")
                ]
                if len(candidates) == 1:
                    sector, mapping = next(iter(candidates.items()))
            sector_known_date = (
                sessions[max(document["available_index"], mapping["available_index"])]
                if mapping is not None
                else None
            )
            for security in document["securities"]:
                if not security["start"] <= current_date <= security["end"]:
                    continue
                method = "dated_fca_ticker"
                candidates = (
                    [ticker_isin[security["ticker"]]]
                    if security["ticker"] in ticker_isin
                    else []
                )
                generic = security["class"] == "SHARES"
                if not security["ticker"] or generic:
                    spelling = legal_spelling(document.get("legal_name", ""))
                    method = "exact_historical_legal_spelling"
                    if not spelling or legal_issuers[spelling] != {cnpj}:
                        continue
                    candidates = name_isins[spelling]
                for isin in candidates:
                    observed = known_security[isin]
                    spec = observed.get("security_spec_base", "")
                    share_class = security["class"]
                    preferred = security.get("preferred_class", "")
                    identity_method = method
                    cls = (
                        "ON"
                        if spec.startswith("ON")
                        else "PN"
                        if spec.startswith("PN")
                        else "UNIT"
                        if spec == "UNT"
                        else None
                    )
                    if not generic and cls is not None:
                        if cls != share_class:
                            continue
                        if cls == "PN":
                            if preferred and spec != "PN" + preferred:
                                continue
                            preferred = spec[2:]
                    if (
                        method == "dated_fca_ticker"
                        and first_seen[isin] > document["receipt"]
                    ):
                        # A ticker may later be reused by a different security.
                        # Only an explicit listing date already disclosed in
                        # this FCA may admit an ISIN born after its receipt.
                        listed_index = bisect.bisect_left(sessions, security["start"])
                        preannounced = (
                            security["start"] >= document["receipt"]
                            and listed_index < len(sessions)
                            and first_seen[isin] == sessions[listed_index]
                        )
                        if not preannounced:
                            continue
                        identity_method = "dated_fca_preannounced_listing"
                    if method == "exact_historical_legal_spelling":
                        if first_seen[isin] > document["receipt"]:
                            continue
                        if generic:
                            if cls not in {"ON", "PN"}:
                                continue
                            share_class = cls
                            preferred = spec[2:] if cls == "PN" else ""
                            identity_method = (
                                "original_generic_shares_exact_legal_spelling"
                            )
                        elif cls != share_class:
                            continue
                        if cls == "PN" and preferred and spec != "PN" + preferred:
                            continue
                    if isin in mapped and mapped[isin]["cnpj"] != cnpj:
                        raise ValueError(
                            f"Ambiguous dated issuer for {isin} at {current_date}"
                        )
                    mapped[isin] = {
                        "date": current_date,
                        "isin": isin,
                        "cnpj": cnpj,
                        "cvm_code": document["cvm_code"],
                        "sector": sector,
                        "sector_label": document.get("sector_label") or None,
                        "sector_known_date": sector_known_date,
                        "sector_mapping_id": mapping["id"] if mapping else None,
                        "class": share_class,
                        "preferred_class": preferred,
                        "unit_composition": security["unit_composition"],
                        "fca_id": document["id"],
                        "identity_known_date": sessions[document["available_index"]],
                        "identity_method": identity_method,
                        "identity_effective_start": max(
                            security["start"], first_seen[isin]
                        ),
                    }
        output.extend(mapped.values())
    schema = {
        "date": pl.Date,
        "isin": pl.String,
        "cnpj": pl.String,
        "cvm_code": pl.String,
        "sector": pl.String,
        "sector_label": pl.String,
        "sector_known_date": pl.Date,
        "sector_mapping_id": pl.String,
        "class": pl.String,
        "preferred_class": pl.String,
        "unit_composition": pl.String,
        "fca_id": pl.String,
        "identity_known_date": pl.Date,
        "identity_method": pl.String,
        "identity_effective_start": pl.Date,
    }
    return pl.DataFrame(output, schema=schema)


def quarter_next(value: date) -> date:
    month = value.month + 3
    year = value.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    return date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)


def year_before(value: date) -> date:
    return (
        date(value.year - 1, value.month + 1, 1) - timedelta(days=1)
        if value.month < 12
        else date(value.year - 1, 12, 31)
    )


def expected_filing_distance(
    current: date,
    latest: date | None,
    receipts: dict[date, date],
    sessions: list[date],
    calendar: dict | None = None,
) -> float | None:
    if latest is None:
        return None
    following = quarter_next(latest)
    previous = year_before(following)
    previous_receipt = receipts.get(previous)
    if previous_receipt is None:
        return None
    expectation = following + (previous_receipt - previous)
    known = sessions
    through = sessions[-1]
    if calendar is not None:
        through = calendar["base_through"]
        if current >= calendar["available_date"]:
            known, through = calendar["full_sessions"], calendar["through"]
    origin = bisect.bisect_left(known, current)
    if expectation <= through:
        return float(bisect.bisect_left(known, expectation) - origin)
    # An unannounced future calendar is an explicitly labelled weekday estimate.
    # Never clip a future expected date to the last date in the research store.
    return float(
        len(known) - origin + np.busday_count(through + timedelta(days=1), expectation)
    )


def header_only_filing_lags(events: list[dict], headers: list[dict]) -> list[dict]:
    """Original headers can date seasonal lags without duplicating RAD clocks."""
    linked = {r["id"] for r in events if r["group"] == "structured" and r.get("id")}
    return [
        {**d, "group": "original_filing_header", "subject": ""}
        for d in headers
        if d["version"] == 1 and d["id"] not in linked
    ]


def event_features(
    events: list[dict], sessions: list[date], calendar: dict | None = None
) -> pl.DataFrame:
    by_issuer = defaultdict(list)
    for event in events:
        index = available_session(event["receipt"], sessions)
        if index < len(sessions):
            by_issuer[event["cvm_code"]].append((index, event))
    output = []
    for code, source in by_issuer.items():
        source.sort(
            key=lambda pair: (
                pair[0],
                pair[1]["receipt"]
                if isinstance(pair[1]["receipt"], datetime)
                else datetime.combine(pair[1]["receipt"], clock_time.max),
            )
        )
        cursor = 0
        last_filing = last_fact = last_dividend = last_offer = None
        last_dfp = None
        fact_indices = []
        receipts = {}
        receipt_indices = {}
        latest_reference = None
        latest_reference_source_index = None
        for index, current in enumerate(sessions):
            while cursor < len(source) and source[cursor][0] <= index:
                event_index, event = source[cursor]
                cursor += 1
                text = normalized(event["kind"] + " " + event.get("subject", ""))
                if event["group"] in ("structured", "original_filing_header"):
                    if event["group"] == "structured":
                        last_filing, last_dfp = (
                            event_index,
                            float(event["kind"].startswith("DFP")),
                        )
                    ref = event["reference"]
                    if ref is not None:
                        if str(event.get("version", "")) == "1":
                            receipts.setdefault(
                                ref,
                                event["receipt"].date()
                                if isinstance(event["receipt"], datetime)
                                else event["receipt"],
                            )
                            receipt_indices.setdefault(ref, event_index)
                        if latest_reference is None or ref > latest_reference:
                            latest_reference = ref
                            latest_reference_source_index = event_index
                if event["group"] == "material_fact":
                    last_fact = event_index
                    fact_indices.append(event_index)
                if any(
                    word in text
                    for word in (
                        "dividendo",
                        "juros sobre capital",
                        "juros sobre o capital",
                        "proventos",
                    )
                ):
                    last_dividend = event_index
                if any(
                    word in text
                    for word in (
                        "recompra",
                        "oferta publica",
                        "oferta de acoes",
                        "distribuicao publica",
                    )
                ):
                    last_offer = event_index
            # Zero counts are observed only inside the completed source-query era.
            record = {
                "date": current,
                "cvm_code": code,
                "sessions_since_financial_filing": float(index - last_filing)
                if last_filing is not None
                else None,
                "filing_is_dfp": last_dfp,
                "sessions_since_material_fact": float(index - last_fact)
                if last_fact is not None
                else None,
                "material_fact_count_20": float(
                    len(fact_indices) - bisect.bisect_left(fact_indices, index - 19)
                ),
                "dividend_announcement_age": float(index - last_dividend)
                if last_dividend is not None
                else None,
                "offering_or_buyback_flag": float(
                    last_offer is not None and index - last_offer < 5
                ),
                "sessions_until_expected_filing": expected_filing_distance(
                    current, latest_reference, receipts, sessions, calendar
                ),
            }
            expected_source_index = None
            if record["sessions_until_expected_filing"] is not None:
                prior_reference = year_before(quarter_next(latest_reference))
                expected_source_index = min(
                    latest_reference_source_index, receipt_indices[prior_reference]
                )
                expectation = quarter_next(latest_reference) + (
                    receipts[prior_reference] - prior_reference
                )
                if (
                    calendar is not None
                    and current >= calendar["available_date"]
                    and expectation > calendar["base_through"]
                ):
                    expected_source_index = min(
                        expected_source_index,
                        bisect.bisect_left(sessions, calendar["available_date"]),
                    )
            ages = {
                "sessions_since_financial_filing": last_filing,
                "filing_is_dfp": last_filing,
                "sessions_since_material_fact": last_fact,
                "material_fact_count_20": index,
                "dividend_announcement_age": last_dividend,
                "offering_or_buyback_flag": index,
                "sessions_until_expected_filing": expected_source_index,
            }
            for feature, source_index in ages.items():
                record[feature + "_age_sessions"] = (
                    float(index - source_index)
                    if source_index is not None and record[feature] is not None
                    else None
                )
            output.append(record)
    return pl.DataFrame(
        output,
        schema={
            "date": pl.Date,
            "cvm_code": pl.String,
            **dict.fromkeys(FEATURES_EVENTS, pl.Float64),
            **{f + "_age_sessions": pl.Float64 for f in FEATURES_EVENTS},
        },
    )


def flow_account(document: dict, basis: str, metric: str) -> dict | None:
    """A missing parent flow needs explicit same-period NCI, never assumed zero."""
    book = document.get("accounts", {}).get(basis, {})
    if metric in book:
        return book[metric]
    if metric == "parent_income" and basis == "con":
        total, minority = book.get("net_income"), book.get("minority_income")
        if (
            total is not None
            and minority is not None
            and all(total[key] == minority[key] for key in ("start", "end"))
        ):
            return {**total, "value": total["value"] - minority["value"]}
    return None


def fiscal_quarters(
    ledger: dict,
    basis: str,
    metric: str,
    source_indices: dict[date, int] | None = None,
    source_versions: dict[date, int] | None = None,
) -> dict[date, float]:
    cumulative = {}
    for document in ledger.values():
        account = flow_account(document, basis, metric)
        if account is None or account["start"] is None:
            continue
        months = (
            (account["end"].year - account["start"].year) * 12
            + account["end"].month
            - account["start"].month
            + 1
        )
        end = account["end"]
        last_day = date(
            end.year + (end.month == 12), end.month % 12 + 1, 1
        ) - timedelta(days=1)
        if months in (3, 6, 9, 12) and account["start"].day == 1 and end == last_day:
            cumulative[(account["start"], months // 3)] = (
                account,
                document.get("available_index", 0),
                document["version"],
            )
    quarters = {}
    for (start, number), (account, source_index, source_version) in cumulative.items():
        prior = cumulative.get((start, number - 1))
        if number == 1 or prior is not None:
            quarters[account["end"]] = account["value"] - (
                prior[0]["value"] if prior else 0
            )
            if source_indices is not None:
                source_indices[account["end"]] = (
                    min(source_index, prior[1]) if prior else source_index
                )
            if source_versions is not None:
                source_versions[account["end"]] = (
                    max(source_version, prior[2]) if prior else source_version
                )
    return quarters


def trailing_twelve_months(
    ledger: dict,
    basis: str,
    metric: str,
    end: date,
    source_versions: list[int] | None = None,
) -> tuple[float | None, int | None]:
    """Annual flow directly, otherwise current YTD + prior annual − prior YTD.

    Every source is its own latest version already public at this decision.
    Fiscal starts, ends and accounting basis must match exactly; missing prior
    standalone quarters do not suppress an independently observed annual TTM.
    """

    def observation(endpoint):
        candidates = [
            d
            for d in ledger.values()
            if d["reference"] == endpoint and flow_account(d, basis, metric) is not None
        ]
        if not candidates:
            return None
        document = max(
            candidates,
            key=lambda d: (d["version"], d.get("available_index", 0), d["id"]),
        )
        account = flow_account(document, basis, metric)
        if account["end"] != endpoint or account["start"] is None:
            return None
        return account, document.get("available_index", 0), document["version"]

    current = observation(end)
    if current is None:
        return None, None
    account, source_index, source_version = current
    start = account["start"]
    months = (end.year - start.year) * 12 + end.month - start.month + 1
    last_day = date(end.year + (end.month == 12), end.month % 12 + 1, 1) - timedelta(
        days=1
    )
    if start.day != 1 or end != last_day:
        return None, None
    if months == 12:
        if source_versions is not None:
            source_versions.append(source_version)
        return account["value"], source_index
    if months not in (3, 6, 9):
        return None, None
    annual_end = start - timedelta(days=1)
    prior_year_end = year_before(end)
    annual, prior_ytd = observation(annual_end), observation(prior_year_end)
    prior_start = start.replace(year=start.year - 1)
    if (
        annual is None
        or prior_ytd is None
        or annual[0]["start"] != prior_start
        or prior_ytd[0]["start"] != prior_start
    ):
        return None, None
    if source_versions is not None:
        source_versions.extend((source_version, annual[2], prior_ytd[2]))
    return account["value"] + annual[0]["value"] - prior_ytd[0]["value"], min(
        source_index, annual[1], prior_ytd[1]
    )


def fundamental_state(ledger: dict, sector: str | None = None) -> dict:
    """All inputs are documents already received at the current decision.

    Missing original contents are not backdated from a later version. Annual
    TTM is directly observed; interim TTM uses a matching known YTD bridge.
    Standalone Q4 and SUE retain their consecutive-quarter requirements.
    SUE uses the eight previous seasonal earnings changes, excluding itself.
    """
    result = dict.fromkeys(FEATURES_FUNDAMENTALS)
    if not ledger:
        return result
    latest = max(ledger.values(), key=lambda d: (d["reference"], d["version"]))
    end = latest["reference"]
    basis = next(
        (
            b
            for b in ("con", "ind")
            if "assets" in latest.get("accounts", {}).get(b, {})
            and "equity" in latest["accounts"][b]
        ),
        None,
    )
    if basis is None:
        return result
    stocks = latest["accounts"][basis]
    assets, equity = stocks["assets"]["value"], stocks["equity"]["value"]
    if assets <= 0:
        return result
    result["liabilities_to_assets"] = 1 - equity / assets
    result["_book_equity"] = (
        equity
        if basis == "ind"
        else stocks["parent_equity"]["value"]
        if "parent_equity" in stocks
        else equity - stocks["minority_equity"]["value"]
        if "minority_equity" in stocks
        else None
    )
    result["_basis_consolidated"] = float(basis == "con")
    result["_reference"] = end
    result["_version"] = latest["version"]
    result["_balance_source_index"] = latest.get("available_index", 0)
    flow_sources = defaultdict(dict)
    flow_versions = defaultdict(dict)
    flow = {
        m: fiscal_quarters(ledger, basis, m, flow_sources[m], flow_versions[m])
        for m in ("net_income", "parent_income")
    }
    ttm_versions = defaultdict(list)
    ttm_values = {
        m: trailing_twelve_months(ledger, basis, m, end, ttm_versions[m])
        for m in ("revenue", "gross_profit", "net_income", "parent_income", "cash_flow")
    }
    ttm = {m: value[0] for m, value in ttm_values.items()}
    source_indices = {
        "liabilities_to_assets": latest.get("available_index", 0),
        "book_to_market": latest.get("available_index", 0),
    }
    restated = {
        "liabilities_to_assets": latest["version"] > 1,
        "book_to_market": latest["version"] > 1,
    }

    def ttm_source(metric, endpoint):
        return (
            ttm_values[metric][1]
            if endpoint == end
            else trailing_twelve_months(ledger, basis, metric, endpoint)[1]
        )

    if ttm["gross_profit"] is not None:
        source_indices["gross_profitability"] = min(
            latest.get("available_index", 0), ttm_source("gross_profit", end)
        )
        restated["gross_profitability"] = latest["version"] > 1 or any(
            v > 1 for v in ttm_versions["gross_profit"]
        )
    earnings_metric = "parent_income" if basis == "con" else "net_income"
    if ttm[earnings_metric] is not None:
        source_indices["earnings_yield_ttm"] = ttm_source(earnings_metric, end)
        restated["earnings_yield_ttm"] = any(
            v > 1 for v in ttm_versions[earnings_metric]
        )
    result["_earnings_ttm"] = ttm[earnings_metric]
    previous_year = year_before(end)
    past_asset_documents = [
        d
        for d in ledger.values()
        if d["reference"] == previous_year
        and "assets" in d.get("accounts", {}).get(basis, {})
    ]
    average_assets = (
        (assets + past_asset_documents[-1]["accounts"][basis]["assets"]["value"]) / 2
        if past_asset_documents
        else None
    )
    sector_text = normalized(sector or "")
    financial = any(
        x in sector_text
        for x in (
            "banco",
            "intermediacao financeira",
            "seguros",
            "seguradoras",
            "credito",
        )
    )
    # The published statement can identify a financial accounting chart before
    # a dated sector cadastre is available; no modern sector backprojection.
    statement_financial = stocks["equity"].get("source_code") in {
        "2.05",
        "2.07",
        "2.08",
    } or any(
        word in normalized(account.get("description", ""))
        for account in stocks.values()
        for word in ("intermediacao financeira", "seguradora", "resseguradora")
    )
    financial = financial or statement_financial
    gross_description = normalized(
        stocks.get("gross_profit", {}).get("description", "")
    )
    # Bank 3.03 is admitted only when the reported account describes financial
    # intermediation. Insurer accounts are not forced into gross-profit semantics.
    gross_comparable = not financial or ("intermedia" in gross_description)
    result["_financial"] = float(financial)
    result["_financial_statement_source_index"] = (
        latest.get("available_index", 0)
        if statement_financial or sector is None
        else None
    )
    if ttm["gross_profit"] is not None and gross_comparable:
        result["gross_profitability"] = ttm["gross_profit"] / assets
    if not financial:
        prior_revenue_versions = []
        prior_revenue = trailing_twelve_months(
            ledger, basis, "revenue", previous_year, prior_revenue_versions
        )[0]
        if ttm["revenue"] is not None and prior_revenue not in (None, 0):
            result["revenue_growth_yoy"] = (ttm["revenue"] - prior_revenue) / abs(
                prior_revenue
            )
            source_indices["revenue_growth_yoy"] = min(
                ttm_source("revenue", end), ttm_source("revenue", previous_year)
            )
            restated["revenue_growth_yoy"] = any(
                v > 1 for v in ttm_versions["revenue"] + prior_revenue_versions
            )
        if (
            ttm["net_income"] is not None
            and ttm["cash_flow"] is not None
            and average_assets
            and average_assets > 0
        ):
            result["accruals_to_assets"] = (
                ttm["net_income"] - ttm["cash_flow"]
            ) / average_assets
            source_indices["accruals_to_assets"] = min(
                latest.get("available_index", 0),
                ttm_source("net_income", end),
                ttm_source("cash_flow", end),
                *[
                    d.get("available_index", 0)
                    for d in ledger.values()
                    if d["reference"] == previous_year
                    and "assets" in d.get("accounts", {}).get(basis, {})
                ],
            )
            restated["accruals_to_assets"] = (
                latest["version"] > 1
                or past_asset_documents[-1]["version"] > 1
                or any(
                    v > 1
                    for v in ttm_versions["net_income"] + ttm_versions["cash_flow"]
                )
            )
    earnings = flow[earnings_metric]
    changes = {
        d: value - earnings[year_before(d)]
        for d, value in earnings.items()
        if year_before(d) in earnings
    }
    prior = sorted(d for d in changes if d < end)[-8:]
    if (
        end in changes
        and len(prior) == 8
        and all(quarter_next(a) == b for a, b in zip(prior, prior[1:]))
        and quarter_next(prior[-1]) == end
    ):
        denominator = float(np.std([changes[d] for d in prior], ddof=1))
        if denominator > 0:
            result["sue"] = changes[end] / denominator
            sue_metric = earnings_metric
            source_indices["sue"] = min(
                flow_sources[sue_metric][d]
                for ending in [*prior, end]
                for d in (ending, year_before(ending))
            )
            restated["sue"] = any(
                flow_versions[sue_metric][d] > 1
                for ending in [*prior, end]
                for d in (ending, year_before(ending))
            )
    result["_feature_source_indices"] = source_indices
    result["_feature_has_later_version"] = restated
    return result


def valuation_market(store: Path) -> dict:
    """Only completed observations are used; retrospective action terms are not.

    DISMES changes are a conservative uncertainty barrier, not an inferred
    share-count adjustment. In particular the store's retrospective action
    resolution mask cannot establish that an old capital count remained valid.
    """
    from brazil_rv.v2.corporate_actions import detect_distribution_changes

    observed = np.load(store / "observed.npy", mmap_mode="r")
    changed = detect_distribution_changes(
        np.load(store / "distribution_number.npy", mmap_mode="r"), observed
    )
    barriers = changed | np.load(store / "detected_split_mask.npy", mmap_mode="r")
    barriers |= np.load(store / "ambiguous_action_mask.npy", mmap_mode="r")
    return {
        "columns": {s: i for i, s in enumerate(np.load(store / "isin_index.npy"))},
        "close": np.load(store / "raw_close.npy", mmap_mode="r"),
        "observed": observed,
        "barrier_prefix": np.vstack(
            [
                np.zeros((1, observed.shape[1]), dtype=np.int32),
                np.cumsum(barriers, axis=0),
            ]
        ),
    }


def issuer_market_cap(
    ledger: dict,
    identity_rows: list[dict],
    index: int,
    sessions: list[date],
    market: dict | None,
    capital_changes: list[dict],
) -> tuple[float | None, int | None]:
    """Sum known non-treasury shares times each class's prior observed close.

    Every positive reported class needs exactly one dated compatible security.
    Aggregate PN counts cannot be allocated across preferred subclasses; units
    cannot substitute for their underlying share prices. A
    share-count snapshot is invalidated by a subsequent known capital event or
    completed-session unit uncertainty, until a post-event count is received.
    """
    if market is None or index == 0:
        return None, None
    known = [d for d in ledger.values() if d.get("shares")]
    if not known:
        return None, None
    document = max(known, key=lambda d: (d["reference"], d["version"]))
    shares = document["shares"]
    classes = [cls for cls, count in shares.items() if count > 0]
    if not classes or any(not np.isfinite(v) or v < 0 for v in shares.values()):
        return None, None
    if any(
        event["available_index"] <= index
        and document["reference"] < event["effective"] <= sessions[index - 1]
        for event in capital_changes
    ):
        return None, None
    first = bisect.bisect_right(sessions, document["reference"])
    prefix = market["barrier_prefix"]
    if first > index:
        return None, None
    total = 0.0
    for share_class in classes:
        candidates = [r for r in identity_rows if r["class"] == share_class]
        if len(candidates) != 1:
            return None, None
        identity = candidates[0]
        if share_class == "PN" and identity.get("preferred_class"):
            return None, None
        column = market["columns"].get(identity["isin"])
        if column is None or not market["observed"][index - 1, column]:
            return None, None
        if prefix[index, column] != prefix[first, column]:
            return None, None
        close = float(market["close"][index - 1, column])
        if not np.isfinite(close) or close <= 0:
            return None, None
        total += shares[share_class] * close
    return total, document["available_index"]


def fundamental_features(
    documents: list[dict],
    rad: list[dict],
    identity: pl.DataFrame,
    sessions: list[date],
    market: dict | None = None,
    capital_changes: list[dict] = (),
) -> tuple[pl.DataFrame, dict]:
    exact = {r["id"]: r for r in rad if r["id"] and r["group"] == "structured"}
    by_issuer = defaultdict(list)
    missing = exact_count = date_count = 0
    for d in documents:
        row = exact.get(d["id"])
        if row is not None:
            if (
                row["cvm_code"] != d["cvm_code"]
                or row["reference"] != d["reference"]
                or (row["version"].isdigit() and int(row["version"]) != d["version"])
            ):
                raise ValueError("RAD/statement identity conflict")
            receipt = row["receipt"]
            exact_count += 1
        else:
            receipt = d["receipt"]
            date_count += 1
        d["available_index"] = available_session(receipt, sessions)
        missing += not bool(d.get("accounts"))
        by_issuer[(d["cnpj"][:8], d["cvm_code"])].append(d)
    output = []
    version_composition = defaultdict(
        lambda: {"first_version_only": 0, "includes_later_version": 0}
    )
    valuation_coverage = defaultdict(lambda: defaultdict(int))
    valuation_issuers = defaultdict(set)
    changes_by_issuer = defaultdict(list)
    for event in capital_changes:
        changes_by_issuer[(event["cnpj"][:8], event["cvm_code"])].append(event)
    for key, rows in identity.partition_by(["cnpj", "cvm_code"], as_dict=True).items():
        cnpj, cvm_code = key
        source = sorted(
            by_issuer[(cnpj[:8], cvm_code)],
            key=lambda d: (d["available_index"], d["version"], int(d["id"])),
        )
        cursor = 0
        ledger = {}
        cached = {}
        latest_index = None
        latest_receipt_version = None
        previous_sector = None
        for date_key, dated in (
            rows.sort("date").partition_by("date", as_dict=True).items()
        ):
            current = date_key[0]
            index = bisect.bisect_left(sessions, current)
            changed = False
            while cursor < len(source) and source[cursor]["available_index"] <= index:
                d = source[cursor]
                cursor += 1
                slot = (d["kind"], d["reference"])
                prior = ledger.get(slot)
                if prior is None or (
                    d["version"],
                    d["available_index"],
                    int(d["id"]),
                ) > (prior["version"], prior["available_index"], int(prior["id"])):
                    ledger[slot] = d
                    changed = True
                    latest_index = d["available_index"]
                    latest_receipt_version = d["version"]
            sector = dated.get_column(
                "sector_label" if "sector_label" in dated.columns else "sector"
            )[0]
            if changed or not cached or previous_sector != sector:
                cached = fundamental_state(ledger, sector)
            previous_sector = sector
            financial_source_indices = (
                [bisect.bisect_left(sessions, dated["identity_known_date"][0])]
                if sector is not None
                else []
            )
            if cached.get("_financial_statement_source_index") is not None:
                financial_source_indices.append(
                    cached["_financial_statement_source_index"]
                )
            identity_rows = dated.to_dicts()
            market_cap, capital_index = issuer_market_cap(
                ledger,
                identity_rows,
                index,
                sessions,
                market,
                changes_by_issuer[(cnpj[:8], cvm_code)],
            )
            capital_document = max(
                (d for d in ledger.values() if d.get("shares")),
                key=lambda d: (d["reference"], d["version"]),
                default=None,
            )
            capital_later_version = (
                capital_document is not None and capital_document["version"] > 1
            )
            positive_classes = sum(
                count > 0
                for count in (capital_document or {}).get("shares", {}).values()
            )
            class_scope = (
                "single_class"
                if positive_classes == 1
                else "multiple_classes"
                if positive_classes > 1
                else "unavailable_or_zero_capital"
            )
            for isin in dated.get_column("isin"):
                record = {
                    "date": current,
                    "isin": isin,
                    **{k: cached.get(k) for k in FEATURES_FUNDAMENTALS},
                }
                record["statement_age_sessions"] = (
                    float(index - latest_index) if latest_index is not None else None
                )
                record["fundamental_financial_flag"] = cached.get("_financial")
                record["fundamental_consolidated_flag"] = cached.get(
                    "_basis_consolidated"
                )
                record["current_balance_version"] = cached.get("_version")
                if market_cap is not None:
                    record["log_market_cap"] = float(np.log(market_cap))
                    for feature, numerator in (
                        ("book_to_market", "_book_equity"),
                        ("earnings_yield_ttm", "_earnings_ttm"),
                    ):
                        if cached.get(numerator) is not None:
                            record[feature] = cached[numerator] / market_cap
                # Ages identify the oldest contributing publication. Daily
                # price revaluation never resets the filing/capital age.
                balance_index = cached.get("_balance_source_index")
                for feature in FEATURES_FUNDAMENTALS:
                    source_index = cached.get("_feature_source_indices", {}).get(
                        feature
                    )
                    if feature in (
                        "log_market_cap",
                        "book_to_market",
                        "earnings_yield_ttm",
                    ):
                        source_index = (
                            min(source_index, capital_index)
                            if source_index is not None and capital_index is not None
                            else capital_index
                        )
                    if feature == "statement_age_sessions":
                        source_index = latest_index
                    record[feature + "_age_sessions"] = (
                        float(index - source_index)
                        if record[feature] is not None and source_index is not None
                        else None
                    )
                record["fundamental_financial_flag_age_sessions"] = (
                    float(index - min(financial_source_indices))
                    if record["fundamental_financial_flag"] is not None
                    else None
                )
                record["fundamental_consolidated_flag_age_sessions"] = (
                    float(index - balance_index)
                    if record["fundamental_consolidated_flag"] is not None
                    and balance_index is not None
                    else None
                )
                record["valuation_available_flag"] = float(market_cap is not None)
                record["valuation_available_flag_age_sessions"] = 0.0
                coverage_key = (current.year, class_scope)
                valuation_coverage[coverage_key]["mapped_name_days"] += 1
                valuation_issuers[(*coverage_key, "mapped")].add((cnpj[:8], cvm_code))
                for feature in (
                    "log_market_cap",
                    "book_to_market",
                    "earnings_yield_ttm",
                ):
                    if record[feature] is not None:
                        valuation_coverage[coverage_key][feature] += 1
                        valuation_issuers[(*coverage_key, feature)].add(
                            (cnpj[:8], cvm_code)
                        )
                for feature in FEATURES_FUNDAMENTALS:
                    if record[feature] is None:
                        continue
                    later = cached.get("_feature_has_later_version", {}).get(
                        feature, False
                    )
                    if feature in (
                        "log_market_cap",
                        "book_to_market",
                        "earnings_yield_ttm",
                    ):
                        later = later or capital_later_version
                    if feature == "statement_age_sessions":
                        later = latest_receipt_version > 1
                    version_composition[(current.year, feature)][
                        "includes_later_version" if later else "first_version_only"
                    ] += 1
                output.append(record)
    return pl.DataFrame(
        output,
        schema={
            "date": pl.Date,
            "isin": pl.String,
            **dict.fromkeys(FEATURES_FUNDAMENTALS, pl.Float64),
            **{f + "_age_sessions": pl.Float64 for f in FEATURES_FUNDAMENTALS},
            "fundamental_financial_flag": pl.Float64,
            "fundamental_financial_flag_age_sessions": pl.Float64,
            "fundamental_consolidated_flag": pl.Float64,
            "fundamental_consolidated_flag_age_sessions": pl.Float64,
            "valuation_available_flag": pl.Float64,
            "valuation_available_flag_age_sessions": pl.Float64,
            "current_balance_version": pl.Int32,
        },
    ), {
        "exact_receipt_documents": exact_count,
        "date_only_receipt_documents": date_count,
        "unrecovered_account_documents": missing,
        "first_version_documents": sum(d["version"] == 1 for d in documents),
        "later_version_documents": sum(d["version"] > 1 for d in documents),
        "feature_version_composition": [
            {"year": year, "feature": feature, **counts}
            for (year, feature), counts in sorted(version_composition.items())
        ],
        "feature_version_rule": "First-version-only means all contributing financial accounts and share counts are version1; any later-version contribution classifies the composite as includes_later_version. Cadastre versions are separate identity/sector provenance.",
        "valuation_coverage_by_class_count": [
            {
                "year": year,
                "class_scope": scope,
                **counts,
                "issuer_counts": {
                    feature: len(valuation_issuers[(year, scope, feature)])
                    for feature in (
                        "mapped",
                        "log_market_cap",
                        "book_to_market",
                        "earnings_yield_ttm",
                    )
                },
            }
            for (year, scope), counts in sorted(valuation_coverage.items())
        ],
        "valuation_status": "all_reported_classes_own_version_shares_separate_prior_observed_closes_no_intervening_known_capital_or_unit_boundary",
    }


def public_float_observations(
    root: Path, rad: list[dict], sessions: list[date]
) -> pl.DataFrame:
    """Actual FRE circulating shares, not issued shares mislabeled free float.

    Retained detail versions enter only at their own receipt. Preferred totals
    are not allocated among multiple classes, and units are not invented from
    their component-share counts. The consumer must match the dated class.
    Data_Ultima_Assembleia dates the reported distribution snapshot; the filing
    reference is only the FRE reference year and cannot date share-unit barriers.
    """
    headers = {d["id"]: d for d in filing_headers(root, "fre")}
    exact = {r["id"]: r for r in rad if r["id"] and r["group"] == "cadastre"}
    rows = []
    for path in annual_paths(root, "fre"):
        year = path.stem[-4:]
        for row in read_csv_member(
            path, f"fre_cia_aberta_distribuicao_capital_{year}.csv"
        ):
            document = headers.get(digits(row["ID_Documento"]))
            if document is None:
                continue
            receipt = (
                exact[document["id"]]["receipt"]
                if document["id"] in exact
                else document["receipt"]
            )
            index = available_session(receipt, sessions)
            if index >= len(sessions):
                continue
            for cls, key in (
                ("ON", "Quantidade_Acoes_Ordinarias_Circulacao"),
                ("PN", "Quantidade_Acoes_Preferenciais_Circulacao"),
            ):
                raw = row.get(key)
                if raw and float(raw) > 0:
                    rows.append(
                        {
                            "date": sessions[index],
                            "cnpj": document["cnpj"],
                            "cvm_code": document["cvm_code"],
                            "class": cls,
                            "free_float_shares": float(raw),
                            "document_id": document["id"],
                            "version": document["version"],
                            "reference": document["reference"],
                            "snapshot_date": date.fromisoformat(
                                row["Data_Ultima_Assembleia"]
                            )
                            if row.get("Data_Ultima_Assembleia")
                            else None,
                        }
                    )
    return pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "cnpj": pl.String,
            "cvm_code": pl.String,
            "class": pl.String,
            "free_float_shares": pl.Float64,
            "document_id": pl.String,
            "version": pl.Int32,
            "reference": pl.Date,
            "snapshot_date": pl.Date,
        },
    )


def capital_change_observations(
    root: Path, rad: list[dict], sessions: list[date]
) -> list[dict]:
    """Known FRE capital changes invalidate earlier capital counts.

    Approval dates lacking a distinct effective date are conservative ambiguity
    boundaries only. They are never used to adjust shares or backdate an event's
    public availability. Zero-share capital-value changes are not unit events.
    """
    headers = {d["id"]: d for d in filing_headers(root, "fre")}
    exact = {r["id"]: r for r in rad if r["id"] and r["group"] == "cadastre"}
    output = []
    for path in annual_paths(root, "fre"):
        year = path.stem[-4:]
        for table, effective_column in (
            ("aumento", "Data_Emissao"),
            ("reducao", "Data_Reducao"),
            ("desdobramento", "Data_Aprovacao"),
        ):
            for row in read_csv_member(
                path, f"fre_cia_aberta_capital_social_{table}_{year}.csv"
            ):
                document = headers.get(digits(row["ID_Documento"]))
                effective = row.get(effective_column) or row.get("Data_Deliberacao")
                if document is None or not effective:
                    continue
                if table == "desdobramento":
                    before = float(
                        row.get("Quantidade_Total_Acoes_Antes_Aprovacao") or 0
                    )
                    after = float(
                        row.get("Quantidade_Total_Acoes_Depois_Aprovacao") or 0
                    )
                    changes_shares = before != after
                else:
                    changes_shares = float(row.get("Quantidade_Total_Acoes") or 0) != 0
                if not changes_shares:
                    continue
                receipt = (
                    exact[document["id"]]["receipt"]
                    if document["id"] in exact
                    else document["receipt"]
                )
                output.append(
                    {
                        "cnpj": document["cnpj"],
                        "cvm_code": document["cvm_code"],
                        "document_id": document["id"],
                        "available_index": available_session(receipt, sessions),
                        "effective": date.fromisoformat(effective),
                        "kind": table,
                        "effective_precision": "approval_bound"
                        if table == "desdobramento"
                        else "reported_effective_or_deliberation_bound",
                    }
                )
    return output


def store_axes_and_identity_observations(
    store: Path,
) -> tuple[list[date], list[str], pl.DataFrame]:
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf8"))
    sessions = np.load(store / "date_index.npy").astype("datetime64[D]").tolist()
    if sessions[-1] > END:
        raise ValueError("Round5 CVM consumer cannot open a later development store")
    isins = np.load(store / "isin_index.npy").tolist()
    sources = []
    for source in manifest["sources"]:
        path = Path(source["path"])
        year = re.search(r"equities_daily_(\d{4})\.parquet$", path.name)
        if year and 2009 <= int(year[1]) <= 2024:
            sources.append(
                pl.scan_parquet(path)
                .select(
                    "trade_date",
                    "ticker",
                    "isin",
                    "issuer_short_name",
                    "security_spec_base",
                )
                .filter(pl.col("trade_date") <= END)
                .collect()
            )
    return (
        sessions,
        isins,
        pl.concat(sources)
        .filter(pl.col("isin").is_in(isins))
        .unique()
        .sort("trade_date"),
    )


def target_issuers(root: Path, observations: pl.DataFrame) -> set[str]:
    """Acquisition scope includes historical exact-name issuers and delistings.

    This only chooses source documents to fetch; decision-date identity still
    applies the stricter contemporaneous receipt/class/uniqueness contract.
    """
    tickers = set(observations.get_column("ticker"))
    legal_names = {
        legal_spelling(n) for n in observations.get_column("issuer_short_name")
    }
    issuers = set()
    for path in annual_paths(root, "fca"):
        year = path.stem[-4:]
        for row in read_csv_member(path, f"fca_cia_aberta_valor_mobiliario_{year}.csv"):
            if (
                row["Codigo_Negociacao"].strip().upper() in tickers
                and row["Mercado"] == "Bolsa"
            ):
                issuers.add(digits(row["CNPJ_Companhia"]))
        for row in read_csv_member(path, f"fca_cia_aberta_geral_{year}.csv"):
            if legal_spelling(row["Nome_Empresarial"]) in legal_names:
                issuers.add(digits(row["CNPJ_Companhia"]))
    return issuers


def build(root: Path, store: Path, output: Path) -> dict:
    """Materialize bounded source-derived families, never mutate a base store."""
    output.mkdir(parents=True, exist_ok=False)
    sessions, isins, observations = store_axes_and_identity_observations(store)
    rad = rad_rows(root)
    cadastre = fca_documents(root, sessions, rad)
    write_json(
        output / "fca_source_manifests.json",
        [d["original_fca_source"] for d in cadastre if "original_fca_source" in d],
    )
    write_json(
        output / "fca_identity_documents.json",
        [
            d
            for d in cadastre
            if d["available_index"] < len(sessions)
            and (
                "securities" in d
                or (d.get("sector_code") is not None and d.get("sector_label"))
            )
        ],
    )
    fca_audit = []
    for year in range(2010, 2025):
        annual = [d for d in cadastre if d["reference"].year == year]
        fca_audit.append(
            {
                "year": year,
                "headers": len(annual),
                "own_details": sum("securities" in d for d in annual),
                "original_details": sum(
                    d.get("metadata_source") == "original_fca_xml" for d in annual
                ),
                "generic_share_documents": sum(
                    any(s["class"] == "SHARES" for s in d.get("securities", []))
                    for d in annual
                ),
                "sector_code_translation_required_documents": [
                    d["id"]
                    for d in annual
                    if "securities" in d and d.get("sector_code") is None
                ],
                "original_class_verification_required": [
                    d["id"]
                    for d in annual
                    if d.get("metadata_source") == "annual_csv"
                    and any(
                        s["class"] in {"ON", "PN"} and not s["ticker"]
                        for s in d.get("securities", [])
                    )
                ],
            }
        )
    write_json(output / "fca_source_coverage.json", fca_audit)
    identity = build_identity(cadastre, observations, sessions, isins)
    identity.write_parquet(output / "identity.parquet")
    public_float_observations(root, rad, sessions).write_parquet(
        output / "free_float_observations.parquet"
    )
    calendar_path = root / "calendar_2025_announced.json"
    calendar = json.loads(calendar_path.read_text(encoding="utf8"))
    if sha256(Path(calendar["source"]["path"])) != calendar["source"]["sha256"]:
        raise ValueError("Announced calendar PDF differs from its source binding")
    for key in ("available_date", "base_through", "through"):
        calendar[key] = date.fromisoformat(calendar[key])
    calendar["full_sessions"] = sessions + [
        date.fromisoformat(d) for d in calendar["sessions"]
    ]
    codes = set(identity.get_column("cvm_code"))
    issuer_events = [
        r for r in rad if r["cvm_code"] in codes and r["group"] != "cadastre"
    ]
    header_lags = header_only_filing_lags(
        issuer_events,
        [
            d
            for d in filing_headers(root, "itr") + filing_headers(root, "dfp")
            if d["cvm_code"] in codes
        ],
    )
    write_json(output / "event_original_header_lags.json", header_lags)
    event_state = event_features(
        [*issuer_events, *header_lags],
        sessions,
        calendar,
    )
    events = (
        identity.select("date", "isin", "cvm_code")
        .join(event_state, on=["date", "cvm_code"], how="left")
        .drop("cvm_code")
    )
    events.write_parquet(output / "events.parquet")
    documents = load_accounts(root, set(identity.get_column("cnpj")))
    from .round5_cvm_capital import load_capital_dispositions

    disposition_path = root / "capital_source_dispositions.json"
    capital_dispositions = load_capital_dispositions(disposition_path, documents)
    recovery_audit = {"attached": 0, "invalid": []}
    original_sources = []
    capital_sources = []
    capital_issues = []
    capital_coverage = defaultdict(lambda: defaultdict(int))
    for document in documents:
        shares_source = None
        path = root / "originals" / document["id"]
        if not document.get("accounts") and (path / "manifest.json").exists():
            try:
                attach_viewer_accounts(document, path)
                recovery_audit["attached"] += 1
                original_sources.append(
                    {
                        "document_id": document["id"],
                        "manifest_path": str(path / "manifest.json"),
                        "manifest_sha256": sha256(path / "manifest.json"),
                    }
                )
            except (ValueError, KeyError) as error:
                recovery_audit["invalid"].append(
                    {"id": document["id"], "reason": str(error)}
                )
        zip_root = root / "original_zips" / document["id"]
        if (not document.get("accounts") or document.get("shares") is None) and (
            zip_root / "manifest.json"
        ).exists():
            from .round5_cvm_xml import original_accounts

            original_manifest = json.loads(
                (zip_root / "manifest.json").read_text(encoding="utf8")
            )
            if sha256(zip_root / "source.zip") != original_manifest["sha256"]:
                raise ValueError(
                    "Original financial ZIP differs from its source manifest"
                )
            parsed = original_accounts(document, zip_root / "source.zip")
            if "capital_issue" in parsed:
                capital_issues.append(
                    {
                        "document_id": document["id"],
                        "source": "original_zip",
                        **parsed["capital_issue"],
                    }
                )
            if not document.get("accounts"):
                document.update(parsed)
                recovery_audit["attached"] += 1
            if document.get("shares") is None and parsed.get("shares") is not None:
                document["shares"] = parsed["shares"]
            if shares_source is None and parsed.get("shares") is not None:
                shares_source = "original_zip"
            original_sources.append(
                {
                    "document_id": document["id"],
                    "manifest_path": str(zip_root / "manifest.json"),
                    "manifest_sha256": sha256(zip_root / "manifest.json"),
                }
            )
        capital_root = root / "capital" / document["id"]
        if document.get("shares") is None and (capital_root / "manifest.json").exists():
            from .round5_cvm_capital import load_capital

            document["shares"] = load_capital(document, capital_root)["shares"]
            shares_source = "capital_html"
            capital_sources.append(
                {
                    "document_id": document["id"],
                    "manifest_path": str(capital_root / "manifest.json"),
                    "manifest_sha256": sha256(capital_root / "manifest.json"),
                }
            )
        # An exact own-note audit can disprove a numerically parseable table.
        # It therefore supersedes that table, not only absent/negative counts.
        if document["id"] in capital_dispositions:
            disposition = capital_dispositions[document["id"]]
            document["shares"] = disposition["shares"]
            shares_source = (
                "source_note_reconciliation"
                if disposition["shares"] is not None
                else "audited_unavailable"
            )
            capital_sources.append(
                {
                    "document_id": document["id"],
                    "manifest_path": str(disposition_path),
                    "manifest_sha256": sha256(disposition_path),
                    "disposition": disposition["disposition"],
                }
            )
        capital_coverage[str(document["reference"].year)][
            shares_source or "unavailable"
        ] += 1
    account_units = apply_account_unit_dispositions(root, documents)
    write_json(output / "original_source_manifests.json", original_sources)
    write_json(output / "capital_source_manifests.json", capital_sources)
    write_json(output / "capital_quantity_issues.json", capital_issues)
    capital_changes = capital_change_observations(root, rad, sessions)
    write_json(output / "capital_change_observations.json", capital_changes)
    fundamentals, audit = fundamental_features(
        documents, rad, identity, sessions, valuation_market(store), capital_changes
    )
    fundamentals.write_parquet(output / "fundamentals.parquet")
    family_tables = {}
    for name, table, columns in (
        ("events", events, FEATURES_EVENTS),
        ("fundamentals", fundamentals, FEATURES_FUNDAMENTALS),
    ):
        coverage = (
            table.with_columns(pl.col("date").dt.year().alias("year"))
            .group_by("year")
            .agg(
                pl.col("isin").n_unique().alias("covered_isins"),
                pl.len().alias("mapped_name_days"),
                *[pl.col(c).is_not_null().sum().alias(c) for c in columns],
            )
            .sort("year")
        )
        coverage.write_parquet(output / f"{name}_coverage.parquet")
        family_tables[name] = coverage.to_dicts()
    manifest = {
        "schema": "ROUND5_CVM_DERIVED_V1",
        "source_root": str(root),
        "source_program_sha256": sha256(Path(__file__)),
        "original_zip_parser_sha256": sha256(
            Path(__file__).with_name("round5_cvm_xml.py")
        ),
        "capital_html_parser_sha256": sha256(
            Path(__file__).with_name("round5_cvm_capital.py")
        ),
        "original_fca_parser_sha256": sha256(
            Path(__file__).with_name("round5_cvm_fca.py")
        ),
        "base_store": str(store),
        "base_store_manifest_sha256": sha256(store / "manifest.json"),
        "through": str(END),
        "availability_rule": "exact RAD minute upper bound (+1min), first15:45at-or-after; date-only nextsession",
        "version_rule": "only own version account contents; missing original versions remain unavailable until recovered or replaced at actual later receipt",
        "identity_rule": "own-receipt FCA cash ticker or unique exact historical legal spelling, prior COTAHIST ISIN and class, bounded existing security; original generic shares require exact-name observed ON/PN, never units or modern HTML class backfill; legal CNPJ root plus CVM registration",
        "sector_rule": "explicit numeric FCA code at own receipt; annual labels resolve at each decision using only already-received unambiguous exact-ID code/label evidence; sector_known_date and sector_mapping_id bind availability and source; future evidence never changes earlier groups; display labels remain own-receipt accounting annotations",
        "identity_rows": identity.height,
        "identity_isins": identity.get_column("isin").n_unique(),
        "receipt_audit": audit,
        "original_recovery": recovery_audit,
        "account_unit_dispositions_source": account_units,
        "header_only_original_lag_documents": len(header_lags),
        "capital_sources_by_reference_year": dict(capital_coverage),
        "capital_dispositions_source": {
            "path": str(disposition_path),
            "sha256": sha256(disposition_path),
        }
        if disposition_path.exists()
        else None,
        "source_contracts": {
            name: {
                "path": str(Path(__file__).parents[3] / "preregistrations" / name),
                "sha256": sha256(Path(__file__).parents[3] / "preregistrations" / name),
            }
            for name in (
                "v2_round5_data.json",
                "v2_round5_event_calendar_amendment.md",
                "v2_round5_valuation_amendment.md",
                "v2_round5_fca_identity_amendment.md",
            )
        },
        "source_manifests": {
            name: {"path": str(root / name), "sha256": sha256(root / name)}
            for name in (
                "annual_manifest.json",
                "rad_manifest.json",
                "calendar_2025_announced.json",
            )
        },
        "age_rule": "filing and capital publication ages carried; multiquarter metrics use oldest source receipt in required accounting window; daily event-window absence has current-decision age zero",
        "coverage": family_tables,
        "files": {
            p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in output.iterdir()
            if p.is_file()
        },
        "limitations": [
            "EarlyFCA2010–2017 omits ticker symbols; only unique exact historical legal spellings with contemporaneous class/existing-security evidence are admitted.",
            "Original FCA versions are recovered at their own receipt; any still-missing original remains unavailable until an actually received supported version. Unresolved annual sector-code translations remain missing rather than mixing labels and codes.",
            "Valuation uses only own-version XML/HTML counts with independent quantity scale, or document-specific audited note reconciliation. Annual capital CSV omits quantity scale and is not a model count source. Negative/unknown treasury is never subtracted or zero-filled; audited unusable counts remain missing. Every positive issued class requires an unambiguous prior observed price; DISMES/known capital boundaries remain enforced. Independently verified action completeness is not established.",
            "Accounting source recipes retain masked bank/insurance incomparable fields.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("acquire", "rad", "inventory", "recover", "build")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.command == "acquire":
        print(json.dumps(acquire_annual(args.root), default=str))
    elif args.command == "rad":
        print(json.dumps(acquire_rad(args.root), default=str))
    elif args.command in ("inventory", "recover"):
        if args.store is None:
            parser.error("--store is required")
        _, _, observations = store_axes_and_identity_observations(args.store)
        issuers = target_issuers(args.root, observations)
        documents = load_accounts(args.root, issuers)
        if args.command == "inventory":
            result = recovery_inventory(documents)
            write_json(args.root / "recovery_inventory.json", result)
            print(
                json.dumps(
                    {k: v for k, v in result.items() if k != "missing"}, default=str
                )
            )
        else:
            print(json.dumps(recover_originals(args.root, documents), default=str))
    else:
        if args.store is None or args.output is None:
            parser.error("--store and --output are required")
        print(json.dumps(build(args.root, args.store, args.output), default=str))


if __name__ == "__main__":
    main()
