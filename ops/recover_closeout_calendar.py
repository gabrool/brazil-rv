"""Selected calendar-source leads; preserve originals and failed receipts."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

from pypdf import PdfReader

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding

PROJECT = Path(__file__).resolve().parents[1]


def main():
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    root = Path(run["root"]) / "closeout_calendar_sources"
    root.mkdir(exist_ok=True)
    snapshot = root / "executed_manual_leads.py"
    assert not snapshot.exists()
    shutil.copyfile(__file__, snapshot)
    selections = [
        (
            "elektro_20160820",
            ".pdf",
            "https://diariooficial.imprensaoficial.com.br/doflash/prototipo/2016/Agosto/20/empresarial/pdf/pg_0016.pdf",
        ),
        (
            "calendar_2016_www_index",
            ".html",
            "https://www.b3.com.br/pt_br/regulacao/oficios-e-comunicados/bm-fbovespa/?palavraChave=Calend%C3%A1rio&dataIni=01%2F12%2F2015&dataFim=15%2F12%2F2015",
        ),
        (
            "b3_131_2015_calendar",
            ".pdf",
            "https://www.b3.com.br/data/files/4D/37/E9/A9/6C4F25103A135D25790D8AA8/OC%20131-2015-DP.pdf",
        ),
        (
            "cblc_sita_manual",
            ".pdf",
            "https://sita.com.br/sita/assets/2021/05/regras-liquidacao-cblc.pdf",
        ),
        (
            "cblc_modal_manual",
            ".pdf",
            "https://www.modalmais.com.br/wp-content/uploads/2022/03/Procedimentos_Operacionais_CBLC1.pdf",
        ),
        (
            "b3_prior_manual",
            ".pdf",
            "https://www.b3.com.br/data/files/E2/10/C1/DF/386BB510CAF42BB5790D8AA8/Manual%20de%20Procedimentos%20Operacionais%20da%20Camara%20BMFBOVESPA.pdf",
        ),
    ]
    for name, suffix, url in selections:
        receipt = root / f"{name}_receipt.json"
        if receipt.exists():
            continue
        tick = perf_counter()
        path = root / (name + suffix)
        result = subprocess.run(
            [
                "curl.exe",
                "--silent",
                "--show-error",
                "--location",
                "--fail",
                "--max-time",
                "30",
                "--output",
                str(path),
                "--write-out",
                "%{url_effective}",
                url,
            ],
            capture_output=True,
        )
        record = dict(
            url=url,
            code=result.returncode,
            resolved_url=result.stdout.decode(),
            stderr=result.stderr.decode(),
            seconds=perf_counter() - tick,
            retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        if result.returncode == 0:
            record.update(binding(path))
            record["bytes"] = path.stat().st_size
            if suffix == ".pdf":
                pages = [p.extract_text() for p in PdfReader(path).pages]
                write_json_atomic(root / f"{name}_text.json", pages)
        write_json_atomic(receipt, record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
