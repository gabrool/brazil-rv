"""Remove redundant intermediate weights already in verified sealed archives."""

import json
from pathlib import Path
from time import perf_counter

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding
from reclaim_research_storage import allocated

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    out = Path(run["scaling_investigation"]["path"]).parent / "storage"
    receipt = out / "historical_redundant_epochs.json"
    assert not receipt.exists()
    rows, archives = [], []
    for name in ("v2_foundation_recovery.json", "v2_decision_complete_recovery.json"):
        recovery_path = PROJECT / "docs" / name
        recovery = json.loads(recovery_path.read_text())
        assert recovery["verified"] is True
        archive = Path(recovery["archive"])
        assert archive.stat().st_size == recovery["bytes"]
        assert sha256_file(archive) == recovery["sha256"]
        inventory_path = Path(recovery["inventory"])
        assert sha256_file(inventory_path) == recovery["inventory_sha256"]
        inventory = json.loads(inventory_path.read_text())
        if isinstance(inventory, list):
            root = Path(recovery["source"]).resolve()
            files = {row["path"]: row for row in inventory}
        else:
            root = Path(inventory["root"]).resolve()
            files = inventory["files"]
        assert root.is_relative_to(
            Path("C:/quant-data/b3/processed/model_runs").resolve()
        )
        before_count = len(rows)
        for path in root.rglob("epoch_*.pt"):
            resolved = path.resolve()
            assert resolved.is_relative_to(root) and path.parent.name == "epochs"
            assert (path.parent.parent / "selected.pt").is_file()
            relative = path.relative_to(root).as_posix()
            saved = files.get(relative)
            if saved is None:
                continue  # Later artifacts outside this archive are never removed.
            assert (
                path.stat().st_size == saved["bytes"]
                and sha256_file(path) == saved["sha256"]
            )
            rows.append(
                dict(
                    path=str(resolved),
                    archive=str(archive),
                    member=relative,
                    bytes=saved["bytes"],
                    sha256=saved["sha256"],
                    allocated=allocated(path),
                )
            )
        archives.append(
            dict(
                recovery=binding(recovery_path),
                archive=str(archive),
                sha256=recovery["sha256"],
                files=len(rows) - before_count,
            )
        )
    report = dict(
        status="verified_before_deletion",
        files=rows,
        archives=archives,
        driver=binding(Path(__file__)),
        verification="Current archives and original inventories match their qualified recovery hashes; current epoch files match those inventories. Reuse the prior complete decompressed-member verification, without repeating it. Selected weights, EMA, parents, forecasts and manifests stay online.",
    )
    write_json_atomic(receipt, report)
    for row in rows:
        path = Path(row["path"])
        assert path.resolve().is_relative_to(
            Path("C:/quant-data/b3/processed/model_runs").resolve()
        )
        path.unlink()
    report.update(
        status="retired_archived_intermediate_epochs",
        files_removed=len(rows),
        allocated_bytes_released=sum(row["allocated"] for row in rows),
        seconds=perf_counter() - tick,
    )
    write_json_atomic(receipt, report)
    current = json.loads(pointer.read_text())
    current["scaling_historical_epoch_retirement"] = binding(receipt)
    write_json_atomic(pointer, current)
    print(json.dumps({k: v for k, v in report.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    main()
