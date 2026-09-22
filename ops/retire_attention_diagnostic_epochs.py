"""Retire archived intermediate epochs while retaining selected fit artifacts."""

import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import zipfile

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from reclaim_research_storage import allocated

PROJECT = Path(__file__).resolve().parents[1]


def main(scope="patience"):
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    assert scope in ("patience", "invalid_width", "matched_refits")
    if scope == "matched_refits":
        recovery_key = "stage_c_refit_recovery"
        output_key = "scaling_matched_refit_epoch_retirement"
        root = Path(run["stage_c_refit_root"])
        physical = Path(run["root"]) / "matched_refit_fit_storage"
        progress = bound_json(binding(root / "refits.json"))
        assert progress["status"] == "complete"
        paths = []
        for record in progress["completed"]:
            manifest = Path(record["manifest"]["path"])
            assert bound_json(record["manifest"])["status"] == "completed"
            paths.extend(sorted((manifest.parent / "epochs").glob("epoch_*.pt")))
        allowed_roots = (root.resolve(), physical.resolve())
        member_prefix = "evidence/data_refits/"
    else:
        plan_key, recovery_key, member_prefix, output_key = (
            (
                "scaling_parent_patience_plan",
                "scaling_diagnostic_recovery",
                "evidence/attention_diagnostics/parent_patience/",
                "scaling_patience_storage",
            )
            if scope == "patience"
            else (
                "stage_d_width_original_plan",
                "stage_d_width_recovery",
                "evidence/capacity_width_original/",
                "scaling_invalid_width_epoch_retirement",
            )
        )
        root = Path(run[plan_key]["path"]).parent.resolve()
        allowed_roots = (root,)
        paths = sorted(root.rglob("epoch_*.pt"))
    recovery = bound_json(run[recovery_key])
    archive = recovery["archive"]
    assert sha256_file(Path(archive["path"])) == archive["sha256"]
    rows = []
    with zipfile.ZipFile(archive["path"]) as z:
        members = json.loads(z.read("members.json"))
        aliases = json.loads(z.read("aliases.json"))
        for path in paths:
            assert any(path.resolve().is_relative_to(p) for p in allowed_roots)
            assert path.parent.name == "epochs"
            assert (path.parent.parent / "selected.pt").exists()
            manifest_path = path.parent.parent / "run_manifest.json"
            if not manifest_path.exists():
                # The excluded compiler-failed wave includes an interrupted F10
                # fit. Its epoch bytes are archived; selected/resume/history stay.
                assert scope == "invalid_width"
                manifest = {"artifacts": {}}
            else:
                manifest = json.loads(manifest_path.read_text())
            if "selected_ema.pt" in manifest["artifacts"]:
                assert (path.parent.parent / "selected_ema.pt").exists()
            member = member_prefix + path.relative_to(root).as_posix()
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
            rows.append(
                dict(path=str(path), member=member, allocated=allocated(path), **saved)
            )
    assert rows
    storage = Path(run["scaling_investigation"]["path"]).parent / "storage"
    out = (
        storage
        / (
            {
                "patience": "patience_epochs.json",
                "invalid_width": "invalid_width_epochs.json",
                "matched_refits": "matched_refit_epochs.json",
            }[scope]
        )
    )
    assert not out.exists()
    report = dict(
        status="verified_before_retirement",
        archive=archive,
        files=rows,
        logical_bytes=sum(r["bytes"] for r in rows),
        freed_allocated_bytes=sum(r["allocated"] for r in rows),
        driver=binding(Path(__file__)),
    )
    write_json_atomic(out, report)
    for row in rows:
        path = Path(row["path"])
        assert any(path.resolve().is_relative_to(p) for p in allowed_roots)
        path.unlink()
    report.update(
        status="retired_verified_redundant_epochs", seconds=perf_counter() - started
    )
    write_json_atomic(out, report)
    run = json.loads(pointer.read_text())
    run[output_key] = binding(out)
    write_json_atomic(pointer, run)
    print(json.dumps({k: v for k, v in report.items() if k != "files"}))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "patience")
