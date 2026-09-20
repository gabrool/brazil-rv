"""Bound original issuer notices for the three exposure-ranked held events."""

import argparse
import base64
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import shutil

from pypdf import PdfReader
import requests

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]
EXTERNAL = {
    "aliansce_merger_protocol": "https://mz-filemanager.s3.amazonaws.com/b5533c1f-4f1b-4d98-a225-6c21f1107240/assembleias-de-acionistas-ago-e-age/c86c8454540bbc69c7d39180b0b7a0a60706825efc7897434c98079041041052/manual_para_participacao_da_age_do_dia_25062019_as_23%3A59.pdf",
    "soma_b3_index": "https://www.b3.com.br/data/files/FE/11/55/95/09BE09105FE89209AC094EA8/OC%20012-2024-VNC%20Tratamento%20Carteiras%20de%20%C3%8Dndices%20da%20B3%20-%20Evento%20de%20Incorpora%C3%A7%C3%A3o%20do%20Grupo%20de%20Moda%20Soma%20pela%20Arezzo_PT.pdf",
    "aliansce_b3_index": "https://www.b3.com.br/data/files/B2/E0/92/49/3B35C610558925C6AC094EA8/OC%20057-2019%20Tratamento%20nas%20Carteiras%20dos%20%C3%8Dndices%20da%20B3%20referente%20%C3%A0%20Reorganiza%C3%A7%C3%A3o%20Societ%C3%A1ria%20e%20Incorpora%C3%A7%C3%A3o.pdf",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocols", nargs="*", default=[])
    parser.add_argument("--supplement-2019-allos", action="store_true")
    parser.add_argument("--external", nargs="*", choices=EXTERNAL, default=[])
    args = parser.parse_args()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    root = Path(run["root"]) / "held_event_sources"
    root.mkdir(exist_ok=True)
    index_path = root / (
        "allos_2019_index.json" if args.supplement_2019_allos else "issuer_index.json"
    )
    if not index_path.exists():
        parent = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())["store"]
        manifest = bound_json(
            {
                "path": str(Path(parent["root"]) / "manifest.json"),
                "sha256": parent["manifest_sha256"],
            }
        )
        prior = bound_json(manifest["metadata"]["round7_repair"]["source_audit"])
        rad_path = (
            Path(prior["cvm_family_manifest"]["path"]).parent.parent
            / "rad_manifest.json"
        )
        rad = json.loads(rad_path.read_text())
        rows, receipts = [], []
        for rec in rad["files"]:
            if rec["year"] not in (2019, 2024) or rec["group"] == "structured":
                continue
            if args.supplement_2019_allos and rec["year"] != 2019:
                continue
            path = Path(rec["path"])
            raw_data = json.loads(path.read_text(encoding="utf8"))["d"]["dados"]
            found = []
            for raw in raw_data.split("$&&*"):
                f = raw.split("$&")
                if len(f) < 10:
                    continue

                def clean(s):
                    return html.unescape(re.sub("<[^>]+>", "", s)).strip()

                issuer = clean(f[1])
                selected = (
                    rec["year"] == 2019
                    and any(
                        v in issuer.upper()
                        for v in ("NATURA", "ALIANSCE", "SONAE SIERRA")
                    )
                ) or (
                    rec["year"] == 2024
                    and any(
                        v in issuer.upper()
                        for v in ("GRUPO DE MODA SOMA", "AREZZO", "AZZAS")
                    )
                )
                if args.supplement_2019_allos:
                    selected = "ALLOS" in issuer.upper()
                if not selected:
                    continue
                protocol = re.search(r"NumeroProtocoloEntrega=(\d+)", raw)
                found.append(
                    dict(
                        cvm=clean(f[0]),
                        issuer=issuer,
                        category=clean(f[2]),
                        subject=clean(f[4]),
                        reference=clean(f[5]),
                        receipt=clean(f[6]),
                        protocol=protocol[1] if protocol else None,
                        raw=raw,
                    )
                )
            if found:
                rows.extend(found)
                receipts.append(rec)
        write_json_atomic(
            index_path,
            {
                "rows": rows,
                "source_receipts": receipts,
                "rad_manifest": binding(rad_path),
                "scope": "Only 2019 Natura/Aliansce/Sonae and 2024 Soma/Arezzo issuer index rows; no numerical source census or held-out consumer read.",
            },
        )
        shutil.copyfile(
            __file__,
            root
            / (
                "executed_allos_index.py"
                if args.supplement_2019_allos
                else "executed_index.py"
            ),
        )
    index = json.loads(index_path.read_text(encoding="utf8"))
    receipts = []
    for protocol in args.protocols:
        matches = [row for row in index["rows"] if row["protocol"] == protocol]
        if not matches:
            raise ValueError(
                "Protocol absent from bounded issuer receipts: " + protocol
            )
        path = root / (protocol + ".pdf")
        receipt_path = root / (protocol + "_receipt.json")
        url = "https://www.rad.cvm.gov.br/ENET/frmExibirArquivoIPEExterno.aspx"
        if not path.exists():
            viewer = requests.get(
                url, params={"NumeroProtocoloEntrega": protocol}, timeout=40
            )
            viewer.raise_for_status()
            (root / (protocol + "_viewer.html")).write_bytes(viewer.content)
            assert 'id="hdnHabilitaCaptcha" value="N"' in viewer.text
            response = requests.post(
                url + "/ExibirPDF",
                json={
                    "codigoInstituicao": "1",
                    "numeroProtocolo": protocol,
                    "token": "",
                    "versaoCaptcha": "",
                },
                timeout=40,
            )
            response.raise_for_status()
            data = base64.b64decode(response.json()["d"], validate=True)
            assert data.startswith(b"%PDF-")
            path.write_bytes(data)
            write_json_atomic(
                receipt_path,
                {
                    "pdf": binding(path),
                    "url": viewer.url,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "issuer_rows": matches,
                },
            )
        text_path = path.with_suffix(".txt")
        if not text_path.exists():
            text_path.write_text(
                "\n".join(page.extract_text() for page in PdfReader(path).pages),
                encoding="utf8",
            )
        receipts.append(
            {
                "protocol": protocol,
                "pdf": binding(path),
                "text": binding(text_path),
                "receipt": binding(receipt_path),
            }
        )
    if args.protocols:
        write_json_atomic(
            root / ("batch_" + "_".join(args.protocols) + ".json"), receipts
        )
    for name in args.external:
        path = root / (name + ".pdf")
        if not path.exists():
            response = requests.get(EXTERNAL[name], timeout=60)
            response.raise_for_status()
            assert response.content.startswith(b"%PDF-")
            path.write_bytes(response.content)
            write_json_atomic(
                root / (name + "_receipt.json"),
                {
                    "pdf": binding(path),
                    "url": EXTERNAL[name],
                    "resolved_url": response.url,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "availability": "Own dated document; retrieval timestamp is not historical first publication proof.",
                },
            )
        text_path = path.with_suffix(".txt")
        if not text_path.exists():
            text_path.write_text(
                "\n".join(page.extract_text() for page in PdfReader(path).pages),
                encoding="utf8",
            )
        print(
            json.dumps(
                {"external": name, "pdf": binding(path), "text": binding(text_path)}
            )
        )
    print(
        json.dumps(
            {
                "root": str(root),
                "index_rows": len(index["rows"]),
                "downloaded_or_reused": receipts,
            }
        )
    )


if __name__ == "__main__":
    main()
