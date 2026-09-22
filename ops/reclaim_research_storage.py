"""Losslessly compress selected completed research artifacts; retain decoded bytes."""

import ctypes
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding

PROJECT = Path(__file__).resolve().parents[1]
GET_SIZE = ctypes.windll.kernel32.GetCompressedFileSizeW
GET_SIZE.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong)]
GET_SIZE.restype = ctypes.c_ulong


def allocated(path):
    high = ctypes.c_ulong()
    low = GET_SIZE(str(path), ctypes.byref(high))
    if low == 0xFFFFFFFF and ctypes.windll.kernel32.GetLastError():
        raise ctypes.WinError()
    return (high.value << 32) + low


def main(mode):
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    out = Path(run["scaling_investigation"]["path"]).parent / "storage"
    plan_path = out / "completed_artifact_compression.json"
    if mode == "plan":
        roots = [Path("C:/quant-data/b3/processed/model_runs"), Path(run["root"])]
        rows = []
        seen = set()
        for root in roots:
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix not in (".npy", ".pkl", ".json"):
                    continue
                resolved = path.resolve()
                if resolved in seen or path.stat().st_size < 1_000_000:
                    continue
                seen.add(resolved)
                # Neither immutable source/store roots nor model weights are included.
                if not any(resolved.is_relative_to(r.resolve()) for r in roots):
                    continue
                size = allocated(path)
                if size < path.stat().st_size * 0.75:
                    continue
                rows.append(
                    dict(path=str(resolved), bytes=path.stat().st_size, allocated=size)
                )
        rows.sort(key=lambda r: r["allocated"], reverse=True)
        assert not plan_path.exists()
        write_json_atomic(
            plan_path,
            dict(status="planned", files=rows, driver=binding(Path(__file__))),
        )
        print(
            json.dumps(
                dict(
                    files=len(rows),
                    allocated=sum(r["allocated"] for r in rows),
                    largest=rows[:12],
                )
            )
        )
        return
    assert mode == "run"
    report = json.loads(plan_path.read_text())
    log = out / "completed_artifact_compression.jsonl"
    assert not log.exists()
    saved = 0
    with log.open("x", encoding="utf-8") as handle:
        for row in report["files"]:
            path = Path(row["path"])
            digest = sha256_file(path)
            before = allocated(path)
            result = subprocess.run(
                ["compact.exe", "/C", "/EXE:LZX", str(path)],
                capture_output=True,
                text=True,
                check=True,
            )
            assert sha256_file(path) == digest
            record = dict(
                **row,
                sha256=digest,
                before=before,
                after=allocated(path),
                stdout=result.stdout,
            )
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            saved += before - record["after"]
    report.update(
        status="decoded_hashes_preserved",
        saved_bytes=saved,
        seconds=perf_counter() - tick,
        log=binding(log),
    )
    write_json_atomic(plan_path, report)
    run["scaling_completed_artifact_compression"] = binding(plan_path)
    write_json_atomic(pointer, run)
    print(json.dumps({k: v for k, v in report.items() if k != "files"}))


if __name__ == "__main__":
    main(sys.argv[1])
