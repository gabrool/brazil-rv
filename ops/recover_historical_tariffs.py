"""Recover only the dated tariff notices selected by the Stage A source review."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
from urllib.request import urlopen

from pypdf import PdfReader

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding


PROJECT = Path(__file__).resolve().parents[1]
SOURCES = {
    "b3_177_2020": "https://www.b3.com.br/data/files/68/50/2D/3E/99E4671059300467AC094EA8/OC%20177-2020%20PRE%20_Modelo_Intermediario_%28PT%29.pdf",
    "b3_014_2022": "https://www.b3.com.br/data/files/21/E4/03/B3/AAF55810F534EB48AC094EA8/OC%20014-2022-VPC%20Depositaria_Compliance%20e%20Jur%C3%ADdico_REVISADO%201.pdf",
    "b3_017_2023": "https://www.b3.com.br/data/files/BB/35/6C/6C/E810B810E9C1AAA8AC094EA8/OC%20017-2023-VPC%20Consolida%C3%A7%C3%A3o%20de%20Regras%20da%20Pol%C3%ADtica%20Tarifa%C3%A7%C3%A3o%20Produtos%20Mercado%20Renda%20Vari%C3%A1vel%20PT.pdf",
}


def main():
    run = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    root = Path(run["root"]) / "historical_tariff_sources"
    root.mkdir(exist_ok=True)
    if not (root / "retrieval_executed.py").exists():
        shutil.copyfile(__file__, root / "retrieval_executed.py")
    receipts = []
    for name, url in SOURCES.items():
        path = root / f"{name}.pdf"
        receipt = root / f"{name}_receipt.json"
        if receipt.exists():
            receipts.append(json.loads(receipt.read_text(encoding="utf8")))
            continue
        tick = perf_counter()
        with urlopen(url, timeout=40) as response:
            content = response.read()
            resolved = response.url
        assert content.startswith(b"%PDF")
        path.write_bytes(content)
        reader = PdfReader(path)
        pages = [p.extract_text() for p in reader.pages]
        write_json_atomic(root / f"{name}_text.json", pages)
        row = dict(
            id=name,
            url=url,
            resolved_url=resolved,
            **binding(path),
            bytes=len(content),
            pages=len(pages),
            retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
            retrieval_seconds=perf_counter() - tick,
        )
        write_json_atomic(receipt, row)
        receipts.append(row)
    write_json_atomic(root / "sources.json", receipts)
    print(json.dumps(receipts), flush=True)
    # Render the selected source pages once; subsequent checks reuse these images.
    for name, pages in {
        "b3_177_2020": [1, 4, 6, 7, 8],
        "b3_014_2022": [1, 2, 3],
        "b3_017_2023": [1, 2, 4],
    }.items():
        for page in pages:
            stem = root / f"{name}_p{page}"
            if stem.with_suffix(".png").exists():
                continue
            subprocess.run(
                [
                    "pdftoppm",
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-scale-to",
                    "1600",
                    "-png",
                    "-singlefile",
                    str(root / f"{name}.pdf"),
                    str(stem),
                ],
                check=True,
                capture_output=True,
            )


if __name__ == "__main__":
    main()
