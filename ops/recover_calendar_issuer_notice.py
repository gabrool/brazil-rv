"""Follow the specific Elektro August2016 erratum lead in existing RAD receipts."""

import argparse
import base64
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
from time import perf_counter

from pypdf import PdfReader
import requests

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol")
    args = parser.parse_args()
    tick = perf_counter()
    run = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    root = Path(run["root"]) / "calendar_issuer_notice"
    root.mkdir(exist_ok=True)
    index_path = root / "index.json"
    if not index_path.exists():
        prior = bound_json(
            binding(Path(run["root"]) / "held_event_sources/issuer_index.json")
        )
        rad = bound_json(prior["rad_manifest"])
        selected, sources = [], []
        for rec in rad["files"]:
            if rec["year"] != 2016 or rec["group"] not in (
                "material_fact",
                "market_communication",
                "shareholder_notice",
            ):
                continue
            raw = bound_json(rec)["d"]["dados"]
            for line in raw.split("$&&*"):
                fields = line.split("$&")
                if len(fields) < 10:
                    continue
                clean = [
                    html.unescape(re.sub("<[^>]+>", "", x)).strip() for x in fields[:7]
                ]
                if "ELEKTRO" not in clean[1].upper() or "/08/2016" not in clean[6]:
                    continue
                protocol = re.search(r"NumeroProtocoloEntrega=(\d+)", line)
                selected.append(
                    dict(
                        cvm=clean[0],
                        issuer=clean[1],
                        category=clean[2],
                        subject=clean[4],
                        reference=clean[5],
                        receipt=clean[6],
                        protocol=None if protocol is None else protocol[1],
                        raw=line,
                    )
                )
            sources.append(rec)
        write_json_atomic(
            index_path,
            dict(
                scope="Only Elektro August2016 notices in three existing2016 RAD groups; a targeted issuer-original follow-up to the already recorded erratum lead, not a repeated B3 calendar search or census",
                rows=selected,
                sources=sources,
                rad_manifest=prior["rad_manifest"],
            ),
        )
        (root / "executed_index.py").write_bytes(Path(__file__).read_bytes())
    index = bound_json(binding(index_path))
    if not args.protocol:
        print(
            json.dumps(
                {
                    "rows": [
                        {k: v for k, v in row.items() if k != "raw"}
                        for row in index["rows"]
                    ],
                    "seconds": perf_counter() - tick,
                },
                ensure_ascii=True,
            )
        )
        return
    row = next(x for x in index["rows"] if x["protocol"] == args.protocol)
    pdf = root / (args.protocol + ".pdf")
    assert not pdf.exists()
    (root / ("executed_" + args.protocol + ".py")).write_bytes(
        Path(__file__).read_bytes()
    )
    url = "https://www.rad.cvm.gov.br/ENET/frmExibirArquivoIPEExterno.aspx"
    response = requests.get(
        url, params={"NumeroProtocoloEntrega": args.protocol}, timeout=35
    )
    viewer = root / (args.protocol + "_viewer.html")
    viewer.write_bytes(response.content)
    response.raise_for_status()
    assert 'id="hdnHabilitaCaptcha" value="N"' in response.text
    document = requests.post(
        url + "/ExibirPDF",
        json={
            "codigoInstituicao": "1",
            "numeroProtocolo": args.protocol,
            "token": "",
            "versaoCaptcha": "",
        },
        timeout=35,
    )
    document.raise_for_status()
    pdf_bytes = base64.b64decode(document.json()["d"], validate=True)
    assert pdf_bytes.startswith(b"%PDF-")
    pdf.write_bytes(pdf_bytes)
    reader = PdfReader(pdf)
    pdf.with_suffix(".txt").write_text(
        "\n\f\n".join(page.extract_text() for page in reader.pages), encoding="utf8"
    )
    receipt = dict(
        index=binding(index_path),
        row=row,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        url=response.url,
        status=response.status_code,
        viewer=binding(viewer),
        pdf=binding(pdf),
        pages=len(reader.pages),
        seconds=perf_counter() - tick,
    )
    write_json_atomic(root / (args.protocol + "_receipt.json"), receipt)
    print(json.dumps(receipt, ensure_ascii=True))


if __name__ == "__main__":
    main()
