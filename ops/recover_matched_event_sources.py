"""Bound issuer notices for corporate transitions exposed by actual model books."""

import argparse
import base64
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import shutil
from time import perf_counter

from pypdf import PdfReader
import requests

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]
# Selection uses exposed security/date neighborhoods, never the sign of a return.
SCOPES = {
    2018: ("VIA VAREJO", "CASAS BAHIA"),
    2020: ("JSL", "SIMPAR", "TIM ", "TIM PARTIC", "TELEFÔNICA", "TELEFONICA"),
    2022: (
        "LOCAMERICA",
        "LOCALIZA",
        "COMPANHIA DE LOCA",
        "SUL AM",
        "SULAM",
        "MODAL",
        "BK BRASIL",
        "ZAMP",
    ),
    2024: ("3R PETROLEUM", "BRAVA"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocols", nargs="*", default=[])
    parser.add_argument("--supplement", action="store_true")
    parser.add_argument("--settlement-index", action="store_true")
    parser.add_argument("--b3-sula", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    root = Path(run["stage_c_root"]) / "event_sources"
    root.mkdir(exist_ok=True)
    if args.b3_sula:
        path = root / "b3_189_2022.pdf"
        assert not path.exists(), "Reuse the saved original"
        url = "https://www.b3.com.br/data/files/FC/F1/79/98/59135810F534EB48AC094EA8/OC%20189-2022%20PRE%20Tratamento%20%C3%8Dndices%20RDOR%20e%20SULA%20(PT).pdf"
        response = requests.get(url, timeout=40)
        response.raise_for_status()
        assert response.content.startswith(b"%PDF-")
        path.write_bytes(response.content)
        path.with_suffix(".txt").write_text(
            "\n".join(p.extract_text() for p in PdfReader(path).pages), encoding="utf8"
        )
        write_json_atomic(
            root / "b3_189_2022_receipt.json",
            dict(
                pdf=binding(path),
                url=url,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                scope="Dated final ratio/effect and reference to general loan manual only; collateral and index dates do not prove shareholder custody or model-history permission.",
                seconds=perf_counter() - started,
            ),
        )
        shutil.copyfile(__file__, root / "executed_b3_sula_retrieval.py")
        print(json.dumps(binding(path)))
        return
    scope = {2022: ("UNIDAS",), 2023: ("REDE D",)} if args.supplement else SCOPES
    label = "supplement" if args.supplement else "issuer"
    if args.settlement_index:
        scope = {2022: ("22691", "24821")}
        label = "settlement"
    index_path = root / (label + "_index.json")
    if not index_path.exists():
        prior = bound_json(
            binding(Path(run["root"]) / "held_event_sources/issuer_index.json")
        )
        rad = bound_json(prior["rad_manifest"])
        write_json_atomic(
            root / (label + "_plan.json"),
            dict(
                exposure=binding(
                    Path(run["stage_c_root"]) / "exposure_audit/manifest.json"
                ),
                scopes=scope,
                rad_manifest=prior["rad_manifest"],
                purpose="Resolve nine actually exposed transition leads together; reuse all previously admitted events and sources. INEP one-day gap is a separate quote disposition, not an assumed action.",
                admissions="No corporate terms, account transfer, issuer history or loan alias admitted by the index. Original dated notices and separate clock/quantity qualification required.",
            ),
        )
        rows, receipts = [], []
        for rec in rad["files"]:
            if rec["year"] not in scope or rec["group"] not in (
                "material_fact",
                "market_communication",
                "shareholder_notice",
            ):
                continue
            selected = []
            for raw in bound_json(rec)["d"]["dados"].split("$&&*"):
                fields = raw.split("$&")
                if len(fields) < 10:
                    continue

                def clean(value):
                    return html.unescape(re.sub("<[^>]+>", "", value)).strip()

                issuer = clean(fields[1])
                selected_issuer = (
                    str(int(re.sub(r"\D", "", clean(fields[0])))) in scope[rec["year"]]
                    if args.settlement_index
                    else any(s in issuer.upper() for s in scope[rec["year"]])
                )
                if not selected_issuer:
                    continue
                protocol = re.search(r"NumeroProtocoloEntrega=(\d+)", raw)
                selected.append(
                    dict(
                        year=rec["year"],
                        cvm=clean(fields[0]),
                        issuer=issuer,
                        category=clean(fields[2]),
                        subject=clean(fields[4]),
                        reference=clean(fields[5]),
                        receipt=clean(fields[6]),
                        protocol=protocol[1] if protocol else None,
                        raw=raw,
                    )
                )
            if selected:
                rows.extend(selected)
                receipts.append(rec)
        write_json_atomic(
            index_path,
            dict(
                rows=rows, source_receipts=receipts, rad_manifest=prior["rad_manifest"]
            ),
        )
        shutil.copyfile(
            __file__,
            root / ("executed_" + label + "_index.py"),
        )
    index = bound_json(binding(index_path))
    sources = []
    for protocol in args.protocols:
        matches = [r for r in index["rows"] if r["protocol"] == protocol]
        assert matches, protocol
        path = root / (protocol + ".pdf")
        receipt = root / (protocol + "_receipt.json")
        if not path.exists():
            url = "https://www.rad.cvm.gov.br/ENET/frmExibirArquivoIPEExterno.aspx"
            viewer = requests.get(
                url, params={"NumeroProtocoloEntrega": protocol}, timeout=40
            )
            viewer.raise_for_status()
            (root / (protocol + "_viewer.html")).write_bytes(viewer.content)
            assert 'id="hdnHabilitaCaptcha" value="N"' in viewer.text
            response = requests.post(
                url + "/ExibirPDF",
                json=dict(
                    codigoInstituicao="1",
                    numeroProtocolo=protocol,
                    token="",
                    versaoCaptcha="",
                ),
                timeout=40,
            )
            response.raise_for_status()
            payload = base64.b64decode(response.json()["d"], validate=True)
            assert payload.startswith(b"%PDF-")
            path.write_bytes(payload)
            write_json_atomic(
                receipt,
                dict(
                    pdf=binding(path),
                    url=viewer.url,
                    issuer_rows=matches,
                    retrieved_at=datetime.now(timezone.utc).isoformat(),
                ),
            )
        txt = path.with_suffix(".txt")
        if not txt.exists():
            txt.write_text(
                "\n".join(p.extract_text() for p in PdfReader(path).pages),
                encoding="utf8",
            )
        sources.append(
            dict(
                protocol=protocol,
                pdf=binding(path),
                text=binding(txt),
                receipt=binding(receipt),
            )
        )
    if sources:
        batch = root / ("batch_" + "_".join(args.protocols) + ".json")
        write_json_atomic(
            batch, dict(sources=sources, seconds=perf_counter() - started)
        )
        shutil.copyfile(__file__, batch.with_suffix(".py"))
    print(
        json.dumps(
            dict(
                index_rows=len(index["rows"]),
                sources=sources,
                seconds=perf_counter() - started,
            )
        )
    )


if __name__ == "__main__":
    main()
