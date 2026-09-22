"""Retire exact extracted recovery-test copies, preserving archives and originals."""

import json
from pathlib import Path
from time import perf_counter
import zipfile

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from reclaim_research_storage import allocated

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    base = Path(run["root"]).resolve()
    out = Path(bound_json(run["scaling_matched_stopping_plan"])["root"]) / "storage"
    out.mkdir(exist_ok=True)
    destination = out / "extracted_recovery_copies.json"
    assert not destination.exists()
    sources, files, skipped, seen = [], [], [], set()
    for key, reference in run.items():
        if (
            not key.endswith("recovery")
            or not isinstance(reference, dict)
            or "sha256" not in reference
        ):
            continue
        report = bound_json(reference)
        location = report.get("restored_root")
        archive = report.get("archive")
        if not location or not isinstance(archive, dict) or "sha256" not in archive:
            continue
        root = Path(location).resolve()
        if root in seen or not root.exists():
            continue
        seen.add(root)
        # A folder called capacity_width_recovery is the actual corrected fit
        # root, not a disposable extract. Only explicit verified restored_root
        # receipts authorize this narrowly bounded deletion.
        assert root.parent == base and "recovery" in root.name
        if not str(report.get("status", "")).startswith("verified"):
            skipped.append(
                dict(root=str(root), reason="Different verification status; retained")
            )
            continue
        source = Path(archive["path"])
        assert (
            source.parent.resolve() == base and sha256_file(source) == archive["sha256"]
        )
        with zipfile.ZipFile(source) as z:
            if "members.json" not in z.namelist():
                skipped.append(
                    dict(root=str(root), reason="Different inventory schema; retained")
                )
                continue
            members = json.loads(z.read("members.json"))
            aliases = (
                json.loads(z.read("aliases.json"))
                if "aliases.json" in z.namelist()
                else {}
            )
            # Reconstructed arrays are retained here. This cleanup needs only
            # direct archived members, without rebuilding any earlier audit.
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                resolved = path.resolve()
                assert not path.is_symlink() and resolved.is_relative_to(root)
                name = path.relative_to(root).as_posix()
                rec = members.get(name)
                if not isinstance(rec, dict) or not {"sha256", "bytes"} <= rec.keys():
                    skipped.append(
                        dict(
                            path=str(path),
                            reason="Not a direct recorded member; retained",
                        )
                    )
                    continue
                stored_name = aliases.get(name, name)
                if stored_name not in z.namelist():
                    skipped.append(
                        dict(path=str(path), reason="Inherited member; retained")
                    )
                    continue
                if (
                    path.stat().st_size != rec["bytes"]
                    or sha256_file(path) != rec["sha256"]
                ):
                    skipped.append(
                        dict(path=str(path), reason="Changed since recovery; retained")
                    )
                    continue
                files.append(
                    dict(
                        path=str(resolved),
                        root=str(root),
                        archive=archive,
                        member=stored_name,
                        sha256=rec["sha256"],
                        bytes=rec["bytes"],
                        allocated=allocated(path),
                    )
                )
        sources.append(dict(receipt=reference, root=str(root), archive=archive))
    # Freeze exact resolved paths and restore locations before deletion. Individual
    # files only; no recursive remove, raw tree traversal or archive destruction.
    write_json_atomic(
        out / "extracted_recovery_plan.json",
        dict(sources=sources, files=files, skipped=skipped),
    )
    saved = 0
    for row in files:
        path, root = Path(row["path"]), Path(row["root"])
        assert path.resolve().is_relative_to(root) and root.parent == base
        assert sha256_file(path) == row["sha256"]
        path.unlink()
        saved += row["allocated"]
    result = dict(
        status="retired_exact_extracted_recovery_copies",
        files=len(files),
        freed_allocated_bytes=saved,
        sources=sources,
        plan=binding(out / "extracted_recovery_plan.json"),
        skipped=len(skipped),
        seconds=perf_counter() - tick,
        scope="Only direct archived members of explicitly verified restored_root folders removed. Each archive and extracted byte hash rechecked. Actual fit roots, canonical originals, accepted stores, source archives, reconstructed views and unrecognized/changed files remain. Restore location for each removed copy is recorded.",
    )
    write_json_atomic(destination, result)
    run = json.loads(pointer.read_text())
    run["scaling_recovery_copy_retirement"] = binding(destination)
    write_json_atomic(pointer, run)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
