"""Retire only byte-verified redundant epochs from the completed patience probe."""

import hashlib
import json
from pathlib import Path
from time import perf_counter
import zipfile

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    root = Path(run["scaling_parent_patience_plan"]["path"]).parent.resolve()
    recovery = bound_json(run["scaling_diagnostic_recovery"])
    archive = recovery["archive"]
    assert sha256_file(Path(archive["path"])) == archive["sha256"]
    rows = []
    with zipfile.ZipFile(archive["path"]) as z:
        members = json.loads(z.read("members.json"))
        aliases = json.loads(z.read("aliases.json"))
        for path in sorted(root.rglob("epoch_*.pt")):
            assert path.resolve().is_relative_to(root) and path.parent.name == "epochs"
            assert (path.parent.parent / "selected.pt").exists()
            manifest = json.loads(
                (path.parent.parent / "run_manifest.json").read_text()
            )
            if "selected_ema.pt" in manifest["artifacts"]:
                assert (path.parent.parent / "selected_ema.pt").exists()
            member = (
                "evidence/attention_diagnostics/parent_patience/"
                + path.relative_to(root).as_posix()
            )
            saved = members[member]
            assert (
                path.stat().st_size == saved["bytes"]
                and sha256_file(path) == saved["sha256"]
            )
            restored = z.read(aliases.get(member, member))
            assert (
                len(restored) == saved["bytes"]
                and hashlib.sha256(restored).hexdigest() == saved["sha256"]
            )
            rows.append(dict(path=str(path), member=member, **saved))
    assert len(rows) == 62
    out = root.parent / "storage" / "patience_epochs.json"
    assert not out.exists()
    report = dict(
        status="verified_before_retirement",
        archive=archive,
        files=rows,
        logical_bytes=sum(r["bytes"] for r in rows),
        driver=binding(Path(__file__)),
    )
    write_json_atomic(out, report)
    for row in rows:
        path = Path(row["path"])
        assert path.resolve().is_relative_to(root)
        path.unlink()
    report.update(
        status="retired_verified_redundant_epochs", seconds=perf_counter() - started
    )
    write_json_atomic(out, report)
    run = json.loads(pointer.read_text())
    run["scaling_patience_storage"] = binding(out)
    write_json_atomic(pointer, run)
    print(json.dumps({k: v for k, v in report.items() if k != "files"}))


if __name__ == "__main__":
    main()
