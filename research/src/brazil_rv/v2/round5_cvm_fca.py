"""Recover exact historical FCA metadata without modern enum-label backfills."""

from __future__ import annotations

import html
import http.cookiejar
import io
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import date
from pathlib import Path
import xml.etree.ElementTree as ET

from lxml import etree

from .round5_cvm import ENET, Tables, digits, normalized, sha256, write_json
from .round5_cvm_capital import _Form, _text


def _document_identity(document: dict) -> dict:
    return {
        key: str(document[key])
        for key in ("id", "cnpj", "cvm_code", "reference", "version", "kind")
    }


def _date(value: str | None, missing: date) -> str:
    return str(missing) if not value or value.startswith("0001-") else value[:10]


def original_fca(document: dict, source: Path) -> dict:
    """Original XML supplies historical values; local IDs are not public IDs.

    Legacy code 1 literally means generic Ações. Its modern HTML rendering can
    say ordinary shares, so the XML description, not that rendering, governs.
    ON/PN for a generic row must come from independent dated B3 observations.
    """
    with zipfile.ZipFile(source) as outer:
        envelope = ET.fromstring(outer.read("FormularioCadastral.xml"))
        expected = {
            "NumeroSequencialDocumento": str(document["id"]),
            "NumeroVersaoDocumento": str(document["version"]),
            "DataReferenciaDocumento": str(document["reference"]),
            "CompanhiaAberta/CodigoCvm": document["cvm_code"],
            "CompanhiaAberta/NumeroCnpjCompanhiaAberta": document["cnpj"],
        }
        for key, value in expected.items():
            actual = envelope.findtext(key, "")
            actual = actual[:10] if key == "DataReferenciaDocumento" else digits(actual)
            if actual != value:
                raise ValueError(f"Original FCA envelope differs at {key}")
        nested = [name for name in outer.namelist() if name.lower().endswith(".fca")]
        if len(nested) != 1:
            raise ValueError("Original FCA has no unique nested source package")
        with zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) as inner:
            general = ET.fromstring(inner.read("FormularioCadastral.xml"))
            securities = ET.fromstring(
                inner.read("ValorMobiliarioMercadoNegociacao.xml")
            )
    for key, value in {
        "CompanhiaAberta/CodigoCvm": document["cvm_code"],
        "CompanhiaAberta/NumeroCnpjCompanhiaAberta": document["cnpj"],
        "Documento/DataReferenciaDocumento": str(document["reference"]),
        "Documento/NumeroVersaoDocumento": str(document["version"]),
    }.items():
        actual = general.findtext(key, "")
        actual = actual[:10] if "DataReferencia" in key else digits(actual)
        same = (
            len(actual) == 14 and actual[:8] == value[:8]
            if key == "CompanhiaAberta/NumeroCnpjCompanhiaAberta"
            else actual == value
        )
        if not same:
            raise ValueError(f"Nested FCA identity differs at {key}")
    result = {
        "source_cnpj": digits(
            general.findtext("CompanhiaAberta/NumeroCnpjCompanhiaAberta", "")
        ),
        "legal_name": general.findtext(
            "CompanhiaAberta/NomeRazaoSocialCompanhiaAberta"
        ),
        "sector_code": general.findtext("CodigoSetorAtividadeEmpresa"),
        "sector": None,
        "sector_label_source": None,
        "securities": [],
        "metadata_source": "original_fca_xml",
    }
    if not result["legal_name"]:
        raise ValueError("Original FCA omits its historical issuer name")
    for security in securities:
        description = security.findtext(
            "ValorMobiliarioNegociado/DescricaoOpcaoDominio", ""
        )
        kind = normalized(description).strip()
        share_class = (
            "SHARES"
            if kind == "acoes"
            else "ON"
            if kind.startswith("acoes ordinarias")
            else "PN"
            if kind.startswith("acoes preferenciais")
            else "UNIT"
            if kind == "units"
            else None
        )
        if share_class is None:
            continue
        market_type = security.findtext("MercadoNegociacao/DescricaoOpcaoDominio", "")
        if normalized(market_type) != "bolsa":
            continue
        preferred_description = security.findtext(
            "ClasseAcao/DescricaoOpcaoDominio", ""
        ).strip()
        preferred = (
            security.findtext("ClasseAcao/SiglaOpcaoDominio", "").strip().upper()
        )
        if not preferred:
            letter = re.fullmatch(
                r"(?:classe\s+)?([a-z])", normalized(preferred_description)
            )
            preferred = letter[1].upper() if letter else preferred_description
        for market in security.findall("MercadosNegociacao/MercadoNegociacao"):
            result["securities"].append(
                {
                    "ticker": security.findtext("CodigoNegociacao", "").strip().upper(),
                    "class": share_class,
                    "preferred_class": preferred,
                    "source_class_description": preferred_description,
                    "unit_composition": security.findtext(
                        "ComposicaoBDRUnit", ""
                    ).strip(),
                    "start": _date(market.findtext("DataInicioRelc"), date.min),
                    "end": _date(market.findtext("DataFimRelc"), date.max),
                    "listing_start": _date(market.findtext("DataInicioRelc"), date.min),
                    "listing_end": _date(market.findtext("DataFimRelc"), date.max),
                    "segment_start": _date(market.findtext("DataInicioNeg"), date.min),
                    "segment_end": _date(market.findtext("DataFimNeg"), date.max),
                    "source_type_description": description,
                    "source_segment_description": market.findtext(
                        "Segmento/DescricaoOpcaoDominio"
                    ),
                    "source_type_code": security.findtext(
                        "ValorMobiliarioNegociado/Dominio/CodigoOpcao"
                    ),
                }
            )
    return result


def _general_html(payload: bytes, document: dict) -> dict:
    table = Tables()
    table.feed(_text(payload))
    fields = {}
    for row in table.rows:
        for offset in range(0, len(row) - 1, 2):
            fields[normalized(row[offset]).rstrip(":").strip()] = row[offset + 1]
    source_cnpj = digits(fields.get("c.n.p.j.", ""))
    if len(source_cnpj) != 14 or source_cnpj[:8] != document["cnpj"][:8]:
        raise ValueError("Exact FCA HTML CNPJ differs")
    if digits(fields.get("codigo cvm", "")) != document["cvm_code"]:
        raise ValueError("Exact FCA HTML CVM registration differs")
    name, sector = fields.get("nome empresarial"), fields.get("setor de atividade")
    if not name:
        raise ValueError("Exact FCA HTML omits its historical issuer name")
    return {"legal_name": name, "sector": sector, "source_cnpj": source_cnpj}


def _generic_html_securities(payload: bytes) -> list[dict]:
    """HTML can establish listed shares; its modern class/segment labels cannot."""
    tree = etree.HTML(_text(payload))
    result = []
    for button in tree.xpath('//button[starts-with(@id,"btnDado_")]'):
        label = "".join(button.itertext()).strip()
        if not normalized(label).startswith("acoes"):
            continue
        identifier = button.get("id").replace("btnDado_", "divDado_")
        for row in tree.xpath(f'//div[@id="{identifier}"]/table/tr'):
            cells = [
                " ".join("".join(cell.itertext()).split()) for cell in row.findall("td")
            ]
            if len(cells) != 7 or normalized(cells[0]) != "bolsa":
                continue

            def parsed(value, missing):
                if not value:
                    return str(missing)
                day, month, year = map(int, value.split("/"))
                return str(date(year, month, day))

            result.append(
                {
                    "ticker": "",
                    "class": "SHARES",
                    "preferred_class": "",
                    "unit_composition": "",
                    "start": parsed(cells[2], date.min),
                    "end": parsed(cells[3], date.max),
                    "listing_start": parsed(cells[2], date.min),
                    "listing_end": parsed(cells[3], date.max),
                    "segment_start": parsed(cells[5], date.min),
                    "segment_end": parsed(cells[6], date.max),
                    "source_type_description": label,
                    "source_type_authority": "HTML equity category only; class unresolved",
                }
            )
    return result


def load_fca(document: dict, destination: Path) -> dict | None:
    """Parse hash-verified exact-version sources with the current read contract.

    Source manifests remain immutable when extraction is corrected; derived
    identity manifests bind the parser version that interprets these bytes.
    """
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf8"))
    if manifest["document"] != _document_identity(document):
        raise ValueError("FCA cache document identity differs")
    for source in manifest["sources"]:
        if sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError("FCA source hash differs")
    metadata = manifest["metadata"]
    if metadata and metadata["metadata_source"] == "original_fca_xml":
        for source in manifest["sources"]:
            path = Path(source["path"])
            if not zipfile.is_zipfile(path):
                continue
            try:
                result = original_fca(document, path)
            except (ValueError, KeyError, ET.ParseError, zipfile.BadZipFile):
                continue
            result.update(
                sector=metadata["sector"],
                sector_label_source=metadata["sector_label_source"],
            )
            return result
        raise ValueError("FCA original metadata has no matching source package")
    return manifest["metadata"]


def recover_fca(
    document: dict, destination: Path, existing_zip: Path | None = None
) -> dict:
    """Cache one original and two tiny exact-viewer pages, preserving failures."""
    if (
        str(document["reference"]) > "2024-12-30"
        or str(document["receipt"])[:10] > "2024-12-30"
    ):
        raise ValueError("FCA request exceeds the development cutoff")
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        load_fca(document, destination)
        return json.loads(manifest_path.read_text(encoding="utf8"))
    destination.mkdir(parents=True, exist_ok=True)
    attempts = destination / "attempts"
    attempts.mkdir(exist_ok=True)
    attempt = attempts / f"{len(list(attempts.iterdir())) + 1:04d}"
    attempt.mkdir()
    sources, failures = [], []
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )

    def request(url, name):
        url = urllib.parse.quote(url, safe="/:?&=+#%")
        with opener.open(
            urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
            timeout=45,
        ) as response:
            body = response.read()
        path = attempt / name
        path.write_bytes(body)
        sources.append(
            {"path": str(path), "sha256": sha256(path), "bytes": len(body), "url": url}
        )
        return body

    metadata = None
    for number in range(1, 2 if existing_zip is not None else 3):
        try:
            if existing_zip is not None:
                original = existing_zip
                sources.append(
                    {
                        "path": str(original),
                        "sha256": sha256(original),
                        "bytes": original.stat().st_size,
                        "role": "reused_original_zip",
                    }
                )
            else:
                url = (
                    "https://www.rad.cvm.gov.br/ENETCONSULTA/frmDownloadDocumento.aspx?"
                    + urllib.parse.urlencode(
                        {
                            "CodigoInstituicao": 1,
                            "NumeroSequencialDocumento": document["id"],
                        }
                    )
                )
                body = request(url, f"original_response_{number}.bin")
                if not zipfile.is_zipfile(io.BytesIO(body)):
                    raise ValueError("Original FCA download is not a ZIP")
                original = attempt / f"original_response_{number}.bin"
            metadata = original_fca(document, original)
            break
        except (
            OSError,
            ValueError,
            KeyError,
            ET.ParseError,
            zipfile.BadZipFile,
        ) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                raise
            failures.append(
                {"stage": "original_zip", "attempt": number, "error": str(error)}
            )
    try:
        url = (
            ENET
            + "frmGerenciaPaginaFRE.aspx?"
            + urllib.parse.urlencode(
                {
                    "NumeroSequencialDocumento": document["id"],
                    "CodigoTipoInstituicao": 1,
                }
            )
        )
        viewer = request(url, "viewer.html")
        form = _Form(viewer)
        expected = {
            "hdnNumeroSequencialDocumento": str(document["id"]),
            "hdnCodigoCvm": document["cvm_code"],
            "hdnCodigoTipoDocumento": "1",
            "hdnHabilitaCaptcha": "N",
        }
        if any(form.fields.get(k) != v for k, v in expected.items()):
            raise ValueError("Exact FCA viewer identity/CAPTCHA differs")
        children = re.findall(r"window.frames\[0\].location='([^']+)'", _text(viewer))
        if len(children) != 1:
            raise ValueError("Exact FCA viewer omitted its unique child")
        child = urllib.parse.urljoin(ENET, html.unescape(children[0]))
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(child).query))
        for key, value in {
            "NumeroSequencialDocumento": str(document["id"]),
            "DataReferencia": str(document["reference"]),
            "Versao": str(document["version"]),
            "CodTipoDocumento": "1",
        }.items():
            if query.get(key) != value:
                raise ValueError(f"Exact FCA child identity differs at {key}")
        literal = _general_html(request(child, "general.html"), document)
        if metadata is not None:
            if normalized(metadata["legal_name"]) != normalized(literal["legal_name"]):
                raise ValueError("Original/HTML historical legal name differs")
            metadata.update(
                sector=literal["sector"],
                sector_label_source="exact_fca_html_with_original_numeric_code",
            )
        else:
            options = re.findall(r'<option[^>]+value="([^"]*)"', _text(viewer))
            relative = next(
                html.unescape(x)
                for x in options
                if x.startswith("frmValorMobiliariocoConsultaNovo.aspx")
            )
            parsed = urllib.parse.urlsplit(relative)
            security_query = dict(urllib.parse.parse_qsl(parsed.query))
            security_query.update(
                {
                    k: v
                    for k, v in query.items()
                    if k
                    in (
                        "Hash",
                        "NumeroSequencialDocumento",
                        "NumeroSequencialRegistroCvm",
                        "CodigoTipoInstituicao",
                        "CodTipoDocumento",
                    )
                }
            )
            security_url = urllib.parse.urljoin(
                ENET, parsed.path + "?" + urllib.parse.urlencode(security_query)
            )
            metadata = {
                **literal,
                "sector_code": None,
                "sector_label_source": "exact_fca_html",
                "metadata_source": "exact_fca_html_generic_equity_only",
                "securities": _generic_html_securities(
                    request(security_url, "securities.html")
                ),
            }
    except (OSError, ValueError, KeyError, StopIteration) as error:
        if isinstance(error, urllib.error.HTTPError) and error.code == 429:
            raise
        failures.append({"stage": "exact_html", "error": str(error)})
    manifest = {
        "schema": "ROUND5_CVM_ORIGINAL_FCA_V1",
        "document": _document_identity(document),
        "metadata": metadata,
        "status": "recovered" if metadata is not None else "unavailable",
        "sources": sources,
        "failures": failures,
        "helper_sha256": sha256(Path(__file__)),
        "source_program_sha256": sha256(Path(sys.argv[0]))
        if Path(sys.argv[0]).is_file()
        else None,
        "attempt_result": str(attempt / "result.json"),
    }
    write_json(attempt / "result.json", manifest)
    if metadata is not None:
        write_json(manifest_path, manifest)
    return manifest
