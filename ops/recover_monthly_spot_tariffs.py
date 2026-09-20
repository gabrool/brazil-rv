"""Retrieve the frozen, bounded monthly TX selection; preserve every response."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
from zipfile import ZipFile

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding

PROJECT = Path(__file__).resolve().parents[1]


def main():
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    source = Path(run["root"]) / "tariff_coverage"
    root = source / "monthly"
    root.mkdir(exist_ok=True)
    selection = json.loads((source / "monthly_tx_selection.json").read_text())
    assert not (root / "plan.json").exists()
    shutil.copyfile(__file__, root / "executed.py")
    write_json_atomic(
        root / "plan.json",
        dict(
            selection=binding(source / "monthly_tx_selection.json"),
            workers=4,
            timeout_seconds=20,
            purpose="Individual retrieval after the bulk HTTP timeout; no successful TX response is repeated.",
        ),
    )

    def fetch(item):
        tick = perf_counter()
        path = root / item["filename"]
        url = (
            "https://www.b3.com.br/pesquisapregao/download?filelist=" + item["filename"]
        )
        row = dict(item, url=url)
        try:
            subprocess.run(
                [
                    "curl.exe",
                    "--silent",
                    "--show-error",
                    "--location",
                    "--fail",
                    "--max-time",
                    "20",
                    "--output",
                    str(path),
                    url,
                ],
                capture_output=True,
                check=True,
            )
            with ZipFile(path) as z:
                members = z.namelist()
            row.update(
                **binding(path),
                bytes=path.stat().st_size,
                members=members,
                status="recovered" if members else "empty_archive",
            )
        except Exception as error:
            row.update(status="retrieval_failed", error=repr(error))
            if isinstance(error, subprocess.CalledProcessError):
                row["stderr"] = error.stderr.decode("utf8", errors="replace")
        row.update(
            retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
            seconds=perf_counter() - tick,
        )
        write_json_atomic(root / (item["filename"] + ".receipt.json"), row)
        return row

    tick = perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(fetch, selection["selection"]))
    report = dict(rows=rows, seconds=perf_counter() - tick)
    write_json_atomic(root / "retrieval.json", report)
    print(
        json.dumps(
            {
                "seconds": report["seconds"],
                "statuses": {
                    s: sum(r["status"] == s for r in rows)
                    for s in sorted({r["status"] for r in rows})
                },
            }
        )
    )


if __name__ == "__main__":
    main()
