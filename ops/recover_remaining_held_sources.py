"""Select new ENAT original receipts; preserve all previously accepted sources."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocols", nargs="*", default=[])
    parser.add_argument("--allos-2020", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    run = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    prior = Path(run["root"]) / "held_event_sources"
    root = Path(run["root"]) / "remaining_held_sources"
    root.mkdir(exist_ok=True)
    index_path = root / (
        "allos_2020_index.json" if args.allos_2020 else "enat_index.json"
    )
    if not index_path.exists():
        earlier = bound_json(binding(prior / "issuer_index.json"))
        rad = bound_json(earlier["rad_manifest"])
        rows, receipts = [], []
        for rec in rad["files"]:
            if rec["year"] != (2020 if args.allos_2020 else 2024) or rec[
                "group"
            ] not in (
                "material_fact",
                "market_communication",
                "shareholder_notice",
            ):
                continue
            raw_data = bound_json(rec)["d"]["dados"]
            selected = []
            for raw in raw_data.split("$&&*"):
                fields = raw.split("$&")
                if len(fields) < 10:
                    continue

                def clean(value):
                    return html.unescape(re.sub("<[^>]+>", "", value)).strip()

                issuer = clean(fields[1])
                issuer_selected = (
                    clean(fields[0]) == "02235-7"
                    if args.allos_2020
                    else any(x in issuer.upper() for x in ("ENAUTA", "BRAVA"))
                )
                if not issuer_selected:
                    continue
                protocol = re.search(r"NumeroProtocoloEntrega=(\d+)", raw)
                selected.append(
                    dict(
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
                rows=rows,
                source_receipts=receipts,
                rad_manifest=earlier["rad_manifest"],
                scope="2020 issuer22357 receipts only"
                if args.allos_2020
                else "2024 ENAT/3R issuer receipts only; current BRAVA archive label is not historical issuer identity.",
            ),
        )
        shutil.copyfile(
            __file__,
            root
            / ("executed_allos_index.py" if args.allos_2020 else "executed_index.py"),
        )
    index = bound_json(binding(index_path))
    sources = []
    for protocol in args.protocols:
        matches = [x for x in index["rows"] if x["protocol"] == protocol]
        if not matches:
            raise ValueError("Protocol absent from bounded original index: " + protocol)
        # Existing accepted originals are dependencies, never copied or overwritten.
        path = prior / (protocol + ".pdf")
        if not path.exists():
            path = root / (protocol + ".pdf")
        receipt_path = path.with_name(protocol + "_receipt.json")
        if not path.exists():
            url = "https://www.rad.cvm.gov.br/ENET/frmExibirArquivoIPEExterno.aspx"
            response = requests.get(
                url, params={"NumeroProtocoloEntrega": protocol}, timeout=40
            )
            response.raise_for_status()
            path.with_name(protocol + "_viewer.html").write_bytes(response.content)
            assert 'id="hdnHabilitaCaptcha" value="N"' in response.text
            pdf_response = requests.post(
                url + "/ExibirPDF",
                json={
                    "codigoInstituicao": "1",
                    "numeroProtocolo": protocol,
                    "token": "",
                    "versaoCaptcha": "",
                },
                timeout=40,
            )
            pdf_response.raise_for_status()
            data = base64.b64decode(pdf_response.json()["d"], validate=True)
            assert data.startswith(b"%PDF-")
            path.write_bytes(data)
            write_json_atomic(
                receipt_path,
                dict(
                    pdf=binding(path),
                    url=response.url,
                    issuer_rows=matches,
                    retrieved_at=datetime.now(timezone.utc).isoformat(),
                ),
            )
        text_path = path.with_suffix(".txt")
        if not text_path.exists():
            text_path.write_text(
                "\n".join(p.extract_text() for p in PdfReader(path).pages),
                encoding="utf8",
            )
        sources.append(
            dict(
                protocol=protocol,
                pdf=binding(path),
                text=binding(text_path),
                receipt=binding(receipt_path),
            )
        )
    if args.protocols:
        batch = root / ("batch_" + "_".join(args.protocols) + ".json")
        write_json_atomic(
            batch, dict(sources=sources, elapsed_seconds=perf_counter() - started)
        )
        shutil.copyfile(__file__, batch.with_suffix(".py"))
    print(
        json.dumps(
            dict(
                index_rows=len(index["rows"]),
                sources=sources,
                elapsed_seconds=perf_counter() - started,
            )
        )
    )


if __name__ == "__main__":
    main()
