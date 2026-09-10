"""Exact-filing capital tables through CVM's public, cookie-bound HTML form."""

from __future__ import annotations

import hashlib
import html
import http.cookiejar
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path

from .round5_cvm import ENET, Tables, net_share_counts, normalized, sha256, write_json


def _text(payload: bytes) -> str:
    return payload.decode("utf-8")


def _identity(document: dict) -> dict:
    return {
        key: str(document[key])
        for key in ("id", "cnpj", "cvm_code", "reference", "version", "kind")
    }


def _request(opener, url: str, data: bytes | None = None) -> bytes:
    # ASP.NET's browser-capability rendering omits normal form postback handlers
    # for an unrecognized User-Agent. CAPTCHA is still checked on every viewer.
    request = urllib.request.Request(
        urllib.parse.quote(url, safe="/:?&=+#%"),
        data=data,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with opener.open(request, timeout=60) as response:
        return response.read()


class _Form(HTMLParser):
    def __init__(self, payload: bytes):
        super().__init__(convert_charrefs=True)
        self.fields: dict[str, str] = {}
        self.groups: list[tuple[str, str]] = []
        self.action = ""
        self.group_handler = ""
        self.select = ""
        self.option: dict | None = None
        self.option_text: list[str] = []
        self.feed(_text(payload))

    def handle_starttag(self, tag, attrs):
        fields = dict(attrs)
        if tag == "form":
            self.action = fields.get("action", "")
        elif tag == "input" and fields.get("name"):
            self.fields[fields["name"]] = fields.get("value", "")
        elif tag == "select":
            self.select = fields.get("name", "")
            if self.select == "cmbGrupo":
                self.group_handler = fields.get("onchange", "")
        elif tag == "option" and self.select:
            self.option, self.option_text = fields, []
            if "selected" in fields:
                self.fields[self.select] = fields.get("value", "")

    def handle_data(self, data):
        if self.option is not None:
            self.option_text.append(data)

    def handle_endtag(self, tag):
        if tag == "option" and self.option is not None:
            if self.select == "cmbGrupo":
                self.groups.append(
                    (self.option["value"], "".join(self.option_text).strip())
                )
            self.option = None
        elif tag == "select":
            self.select = ""


def _viewer(payload: bytes, document: dict) -> tuple[_Form, str, dict]:
    form = _Form(payload)
    if form.fields.get("hdnHabilitaCaptcha") != "N":
        raise ValueError("Capital viewer CAPTCHA is required or its state is unknown")
    expected = {
        "hdnNumeroSequencialDocumento": str(document["id"]),
        "hdnCodigoCvm": str(document["cvm_code"]),
        "hdnCodigoTipoDocumento": {"ITR": "3", "DFP": "4"}[document["kind"]],
    }
    if any(form.fields.get(key) != value for key, value in expected.items()):
        raise ValueError("Capital viewer document identity differs")
    matches = re.findall(r"window.frames\[0\].location='([^']+)'", _text(payload))
    if len(matches) != 1:
        raise ValueError("Capital viewer omitted its exact-filing child URL")
    child = urllib.parse.urljoin(ENET, html.unescape(matches[0]))
    query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(child).query))
    for key, value in {
        "NumeroSequencialDocumento": str(document["id"]),
        "DataReferencia": str(document["reference"]),
        "Versao": str(document["version"]),
        "CodTipoDocumento": expected["hdnCodigoTipoDocumento"],
    }.items():
        if query.get(key) != value:
            raise ValueError(f"Capital child document identity differs at {key}")
    bound = {
        key: query.get(key, "")
        for key in ("Hash", "NumeroSequencialRegistroCvm", "CodigoTipoInstituicao")
    }
    if not all(bound.values()) or bound["Hash"] != form.fields.get("hdnHash"):
        raise ValueError("Capital viewer lacks its bound public session parameters")
    if not child.startswith(ENET):
        raise ValueError("Capital viewer returned a child outside the official ENET")
    return form, child, bound


def parse_capital(payload: bytes, document: dict) -> dict:
    """Paid-in shares less own-period treasury, preserving printed quantity units.

    These are ON/PN totals, not public float or a preferred-subclass/unit mapping.
    A current display name is deliberately not a historical identity attribute.
    Receipt-time admission remains the caller's existing RAD contract.
    """
    table = Tables()
    table.feed(_text(payload))
    rows = table.rows
    if len(rows) != 9 or len(rows[0]) != 2:
        raise ValueError("Capital page lacks the expected two own-period sections")
    unit = normalized(rows[0][0])
    scales = {"numero de acoes (mil)": 1000, "numero de acoes (unidade)": 1}
    if unit not in scales:
        raise ValueError("Capital page quantity scale is unknown")
    reference = datetime.strptime(rows[0][1], "%d/%m/%Y").date().isoformat()
    if reference != str(document["reference"]):
        raise ValueError("Capital table reference date differs")
    scale = scales[unit]
    result = {"reference": reference, "quantity_scale": scale}
    for offset, heading, key in (
        (1, "do capital integralizado", "paid_in"),
        (5, "em tesouraria", "treasury"),
    ):
        if [normalized(cell) for cell in rows[offset]] != [heading]:
            raise ValueError("Capital section labels differ")
        values = {}
        for row, label, code in zip(
            rows[offset + 1 : offset + 4],
            ("ordinarias", "preferenciais", "total"),
            ("ON", "PN", "total"),
            strict=True,
        ):
            if len(row) != 2 or normalized(row[0]) != label:
                raise ValueError("Capital class labels differ")
            if row[1] == "":
                values[code] = None
                continue
            if not re.fullmatch(r"-?(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d+)?", row[1]):
                raise ValueError("Capital quantity is missing or not a printed number")
            values[code] = float(
                Decimal(row[1].replace(".", "").replace(",", ".")) * scale
            )
        # Retain the printed total. Independently rounded thousands can differ
        # from the sum of classes; that does not erase the observed class counts.
        result[key] = values
    result["shares"] = net_share_counts(result["paid_in"], result["treasury"])
    if result["shares"] is None:
        raise ValueError(
            "Capital source has negative, inconsistent or unreported class quantities"
        )
    return result


def load_capital(document: dict, destination: Path) -> dict:
    """Verify the sealed source chain and reparse before attaching any shares."""
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    if manifest["document"] != _identity(document):
        raise ValueError("Archived capital filing identity differs")
    payloads = {}
    for source in manifest["sources"]:
        path = destination / source["file"]
        if sha256(path) != source["sha256"]:
            raise ValueError(f"Archived capital source hash differs: {path}")
        payloads[source["role"]] = path.read_bytes()
    _, _, original_bound = _viewer(payloads["viewer"], document)
    _, child, selected_bound = _viewer(payloads["group"], document)
    if selected_bound != original_bound:
        raise ValueError("Archived capital group changed its bound filing parameters")
    if urllib.parse.urlsplit(child).path != "/ENET/frmDadosComposicaoCapitalITR.aspx":
        raise ValueError("Archived group did not select a capital page")
    page = next(source for source in manifest["sources"] if source["role"] == "capital")
    if page["url"] != child:
        raise ValueError("Archived capital URL differs from the returned child")
    result = parse_capital(payloads["capital"], document)
    if result != manifest["capital"]:
        raise ValueError("Archived capital values differ from source table")
    return result


def load_capital_dispositions(path: Path, documents: list[dict]) -> dict[str, dict]:
    """Read source-audited note reconciliations or genuinely unusable counts.

    Explicit document identities prevent a later note from repairing an earlier
    filing. Evidence bytes and the disposition file are separately hash-bound
    by the final family; no sign convention is inferred for other documents.
    """
    if not path.exists():
        return {}
    by_id = {str(d["id"]): d for d in documents}
    output = {}
    for record in json.loads(path.read_text(encoding="utf8"))["documents"]:
        identifier = record["document"]["id"]
        if identifier not in by_id:
            continue
        if identifier in output or record["document"] != _identity(by_id[identifier]):
            raise ValueError(
                "Capital disposition has a repeated or different filing identity"
            )
        if not record["reason"] or not record["evidence"]:
            raise ValueError("Capital disposition lacks its source findings")
        for source in record["evidence"]:
            if sha256(Path(source["path"])) != source["sha256"]:
                raise ValueError("Capital disposition source evidence hash differs")
        if record["disposition"] == "reconciled":
            shares = net_share_counts(
                record["paid_in_shares"], record["treasury_shares"]
            )
            if shares is None:
                raise ValueError(
                    "Reconciled capital still has unusable class quantities"
                )
        elif record["disposition"] == "audited_unavailable":
            shares = None
        else:
            raise ValueError("Capital disposition does not state a supported outcome")
        output[identifier] = {**record, "shares": shares}
    return output


def capital_page(document: dict, destination: Path) -> dict:
    """Fetch only the exact-filing capital table using the site's normal postback.

    Three small responses replace a multi-MB original ZIP. Failed attempts retain
    their bytes separately; completed per-document manifests are immutable and
    verified on reuse. This routine does not acquire anything after 2024.
    """
    if (
        str(document["reference"]) > "2024-12-30"
        or str(document["receipt"])[:10] > "2024-12-30"
    ):
        raise ValueError("Capital filing lies outside the development admission bound")
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        load_capital(document, destination)
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    attempts = destination / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = attempts / f"{len(list(attempts.iterdir())) + 1:04d}"
    attempt.mkdir()
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
    sources = []

    def retain(role: str, body: bytes, source_url: str, method: str = "GET"):
        path = attempt / (role + (".txt" if role == "post_form" else ".html"))
        path.write_bytes(body)
        sources.append(
            {
                "role": role,
                "file": path.relative_to(destination).as_posix(),
                "url": source_url,
                "method": method,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        )

    body = _request(opener, url)
    retain("viewer", body, url)
    form, _, bound = _viewer(body, document)
    groups = [
        value for value, label in form.groups if normalized(label) == "dados da empresa"
    ]
    if (
        len(groups) != 1
        or "__doPostBack" not in form.group_handler
        or "cmbGrupo" not in form.group_handler
    ):
        raise ValueError("Capital viewer omitted its public group-selection handler")
    action = urllib.parse.urljoin(url, form.action)
    if action != url:
        raise ValueError("Capital postback form action changed the document route")
    fields = {
        **form.fields,
        "cmbGrupo": groups[0],
        "__EVENTTARGET": "cmbGrupo",
        "__EVENTARGUMENT": "",
    }
    data = urllib.parse.urlencode(fields).encode("utf-8")
    retain("post_form", data, action, "POST")
    body = _request(opener, action, data)
    retain("group", body, action, "POST")
    _, child, selected_bound = _viewer(body, document)
    if selected_bound != bound:
        raise ValueError("Capital group changed its bound filing parameters")
    if urllib.parse.urlsplit(child).path != "/ENET/frmDadosComposicaoCapitalITR.aspx":
        raise ValueError("Selected group did not return its capital child")
    body = _request(opener, child)
    retain("capital", body, child)
    capital = parse_capital(body, document)
    manifest = {
        "schema": "ROUND5_CVM_EXACT_CAPITAL_HTML_V1",
        "user_agent": "Mozilla/5.0",
        "document": _identity(document),
        "sources": sources,
        "capital": capital,
        "source_program_sha256": sha256(Path(__file__)),
        "parser_dependency_sha256": sha256(Path(__file__).with_name("round5_cvm.py")),
        "availability": "Exact document's original RAD receipt; retrieval time and display name are not historical features",
    }
    pending = destination / "manifest.pending.json"
    write_json(pending, manifest)
    pending.replace(manifest_path)
    return manifest
