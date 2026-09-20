"""Recover selected original notices for the consolidated historical cost contract."""

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    root = Path(run["root"]) / "tariff_coverage"
    root.mkdir(exist_ok=True)
    plan = json.loads(args.plan.read_text(encoding="utf8"))
    snapshot = root / (args.plan.stem + "_executed.py")
    if not snapshot.exists():
        shutil.copyfile(__file__, snapshot)
    for item in plan:
        receipt = root / (item["id"] + "_receipt.json")
        if receipt.exists():
            print(json.dumps({"reused": item["id"]}), flush=True)
            continue
        tick = perf_counter()
        try:
            path = root / (item["id"] + item["suffix"])
            result = subprocess.run(
                [
                    "curl.exe",
                    "--silent",
                    "--show-error",
                    "--location",
                    "--fail",
                    "--max-time",
                    "35",
                    "--output",
                    str(path),
                    "--write-out",
                    "%{url_effective}",
                    item["url"],
                ],
                capture_output=True,
                check=True,
            )
            content, resolved = path.read_bytes(), result.stdout.decode("utf8")
            if item["suffix"] == ".pdf":
                assert content.startswith(b"%PDF"), "response is not a PDF"
            row = dict(item, **binding(path), bytes=len(content), resolved_url=resolved)
            if item["suffix"] == ".pdf":
                pages = [page.extract_text() for page in PdfReader(path).pages]
                write_json_atomic(root / (item["id"] + "_text.json"), pages)
                row["pages"] = len(pages)
            row["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
            row["seconds"] = perf_counter() - tick
            write_json_atomic(receipt, row)
            print(json.dumps(row), flush=True)
        except Exception as error:
            row = dict(item, error=repr(error), seconds=perf_counter() - tick)
            if isinstance(error, subprocess.CalledProcessError):
                row["stderr"] = error.stderr.decode("utf8", errors="replace")
            write_json_atomic(
                root / (item["id"] + "_" + args.plan.stem + "_failure.json"), row
            )
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
