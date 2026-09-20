"""Selected operative custody notices and the explicitly rejected later draft."""

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
    "b3_189_2024": (
        "https://www.b3.com.br/data/files/AC/A3/86/DB/5940491029BEEC39AC094EA8/OC%20189-2024%20PRE%20Politica%20de%20Tarifacao%20da%20Central%20Depositaria%20de%20Renda%20Variavel.pdf",
        [1],
    ),
    "b3_221_2023": (
        "https://www.b3.com.br/data/files/DE/67/87/42/170BC8103152D4C8AC094EA8/OC%20221-2023%20PRE%20Politica%20de%20Tarifacao%20da%20Central%20Depositaria%20de%20Renda%20Variavel%20B3%20%28PT%29.pdf",
        [1, 2, 3, 4],
    ),
    "b3_041_2024_pre": (
        "https://www.b3.com.br/data/files/D3/32/E8/67/7E48E810C54843E8DC0D8AA8/OC%20041-2024%20PRE%20Politica%20de%20Tarifacao%20da%20Central%20Depositaria%20de%20Renda%20Variavel%20%28PT%29.pdf",
        [1, 2, 4],
    ),
    "b3_041_2024_draft": (
        "https://www.b3.com.br/data/files/3B/F7/86/E9/CD3529106EEC8429AC094EA8/CE%20041-2024-VPC%20NTE_Ajuste_PT.pdf",
        [1, 2],
    ),
}


def main():
    run = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    root = Path(run["root"]) / "custody_tariff_sources"
    root.mkdir(exist_ok=True)
    if not (root / "executed.py").exists():
        shutil.copyfile(__file__, root / "executed.py")
    poppler = (
        Path.home()
        / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe"
    )
    receipts = []
    for name, (url, selected) in SOURCES.items():
        pdf, receipt = root / f"{name}.pdf", root / f"{name}_receipt.json"
        if not receipt.exists():
            tick = perf_counter()
            with urlopen(url, timeout=40) as response:
                content, resolved = response.read(), response.url
            assert content.startswith(b"%PDF")
            pdf.write_bytes(content)
            pages = [p.extract_text() for p in PdfReader(pdf).pages]
            write_json_atomic(root / f"{name}_text.json", pages)
            write_json_atomic(
                receipt,
                dict(
                    id=name,
                    url=url,
                    resolved_url=resolved,
                    **binding(pdf),
                    bytes=len(content),
                    pages=len(pages),
                    retrieved_at=datetime.now(timezone.utc).isoformat(),
                    seconds=perf_counter() - tick,
                ),
            )
        receipts.append(json.loads(receipt.read_text(encoding="utf8")))
        for page in selected:
            stem = root / f"{name}_p{page}"
            if not stem.with_suffix(".png").exists():
                result = subprocess.run(
                    [
                        str(poppler),
                        "-f",
                        str(page),
                        "-l",
                        str(page),
                        "-scale-to",
                        "1400",
                        "-png",
                        "-singlefile",
                        str(pdf),
                        str(stem),
                    ],
                    capture_output=True,
                    check=True,
                )
                (root / f"{name}_p{page}_render_stderr.txt").write_bytes(result.stderr)
    write_json_atomic(root / "boundary_sources.json", receipts)
    print(json.dumps(receipts))


if __name__ == "__main__":
    main()
