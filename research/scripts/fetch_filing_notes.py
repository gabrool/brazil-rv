"""Recover exact-filing note PDFs when CVM's complete-ZIP route fails."""

import argparse
import base64
import http.cookiejar
import html
import io
import json
import pickle
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

from pypdf import PdfReader

from brazil_rv.v2.round5_cvm import normalized, sha256
from brazil_rv.v2.round5_cvm_capital import _Form, _request, _viewer


class EmbeddedPDF(HTMLParser):
    payload = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and attrs.get("name") == "hdnConteudoArquivo":
            self.payload = base64.b64decode(attrs["value"], validate=True)


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--ids", nargs="+", required=True)
args = parser.parse_args()
documents = {
    d["id"]: d
    for d in pickle.loads((args.root / "accepted_documents.pkl").read_bytes())
}


def recover(identifier):
    folder = args.root / "filing_evidence" / identifier
    folder.mkdir(parents=True, exist_ok=True)
    document = documents[identifier]
    sources = []

    def retain(name, data, url):
        path = folder / name
        path.write_bytes(data)
        sources.append({"path": str(path), "sha256": sha256(path), "url": url})

    result = {
        "document": {
            k: str(document[k])
            for k in ("id", "cnpj", "cvm_code", "reference", "version", "kind")
        }
    }
    try:
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )
        url = (
            "https://www.rad.cvm.gov.br/ENET/frmGerenciaPaginaFRE.aspx?"
            + urllib.parse.urlencode(
                {"NumeroSequencialDocumento": identifier, "CodigoTipoInstituicao": 1}
            )
        )
        body = _request(opener, url)
        retain("notes_viewer.html", body, url)
        form, _, bound = _viewer(body, document)
        group = next(
            v for v, label in form.groups if normalized(label) == "notas explicativas"
        )
        action = urllib.parse.urljoin(url, form.action)
        if action != url:
            raise ValueError("note postback changed filing route")
        fields = {
            **form.fields,
            "cmbGrupo": group,
            "__EVENTTARGET": "cmbGrupo",
            "__EVENTARGUMENT": "",
        }
        post = urllib.parse.urlencode(fields).encode()
        retain("notes_post.txt", post, action)
        body = _request(opener, action, post)
        retain("notes_group.html", body, action)
        selected = _Form(body)
        if selected.fields.get("hdnHabilitaCaptcha") != "N" or any(
            selected.fields.get(k) != form.fields.get(k)
            for k in (
                "hdnNumeroSequencialDocumento",
                "hdnCodigoCvm",
                "hdnCodigoTipoDocumento",
            )
        ):
            raise ValueError(
                "note selection changed filing identity or requires CAPTCHA"
            )
        children = re.findall(
            r"window.frames\[0\].location='([^']+)'", body.decode("utf-8")
        )
        if len(children) != 1:
            raise ValueError("note viewer omitted its document child")
        child = urllib.parse.urljoin(url, html.unescape(children[0]))
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(child).query))
        if not child.startswith(
            "https://www.rad.cvm.gov.br/ENET/frmExibirArquivoFRE.aspx?"
        ) or (
            query.get("NumeroSequencialDocumento"),
            query.get("CodigoGrupo"),
            query.get("Tipo"),
        ) != (identifier, "193", "PDF"):
            raise ValueError("note child does not identify this exact filing's notes")
        body = _request(opener, child)
        retain("notes_embedded.html", body, child)
        parser = EmbeddedPDF()
        parser.feed(body.decode("utf-8-sig"))
        if not parser.payload or not parser.payload.startswith(b"%PDF"):
            raise ValueError("exact note viewer returned no PDF")
        retain("notes.pdf", parser.payload, child)
        pages = [
            p.extract_text() or "" for p in PdfReader(io.BytesIO(parser.payload)).pages
        ]
        retain(
            "notes_pages.json",
            json.dumps(pages, ensure_ascii=False).encode("utf-8"),
            child,
        )
        result.update(
            status="extracted", sources=sources, bound=bound, pages=len(pages)
        )
    except Exception as exc:
        result.update(
            status="unresolved", error=f"{type(exc).__name__}: {exc}", sources=sources
        )
    (folder / "notes_evidence.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return {"id": identifier, "status": result["status"], "error": result.get("error")}


with ThreadPoolExecutor(max_workers=3) as pool:
    for result in pool.map(recover, args.ids):
        print(json.dumps(result), flush=True)
