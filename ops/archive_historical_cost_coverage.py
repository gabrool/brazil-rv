"""Restore/hash-check new tariff and corporate evidence, retaining prior dependencies."""

import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter
import zipfile

import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    accepted = bound_json(run["historical_cost_acceptance"])
    base = Path(run["root"])
    root = base / "tariff_coverage"
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    archive = base / f"historical_cost_coverage_{commit[:7]}.zip"
    restored = base / f"historical_cost_recovery_{commit[:7]}"
    assert not archive.exists()
    restored.mkdir(exist_ok=False)
    dependencies = accepted["prior_recovery_dependencies"]
    for record in dependencies.values():
        bound_json(record)
    members, unique, aliases = {}, {}, {}
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:

        def add(name, data):
            assert name not in members
            digest = hashlib.sha256(data).hexdigest()
            members[name] = dict(sha256=digest, bytes=len(data))
            if digest in unique:
                aliases[name] = unique[digest]
            else:
                unique[digest] = name
                z.writestr(name, data)

        paths = subprocess.check_output(
            ["git", "ls-files", "research"], cwd=PROJECT, text=True
        ).splitlines()
        paths += [
            "PROJECT_CONTEXT.md",
            "docs/v2_STAGE_A_CLOSEOUT.md",
            "docs/v2_HISTORICAL_COST_COVERAGE.md",
            "docs/v2_historical_cost_coverage_acceptance.json",
            "docs/v2_economic_data_scaling_run.json",
            "docs/v2_economic_data_scaling_progress.md",
            "ops/verify_corporate_replay.py",
            "ops/audit_custody_fees.py",
            "ops/audit_historical_spot.py",
            "ops/qualify_custody_fees.py",
            "ops/recover_monthly_spot_tariffs.py",
            "ops/inspect_monthly_spot_tariffs.py",
            "ops/recover_tariff_coverage.py",
            "ops/qualify_historical_tariffs.py",
            "ops/audit_tariff_coverage.py",
            "ops/qualify_tariff_coverage.py",
            "ops/qualify_tariff_runtime.py",
            "ops/audit_corporate_custody_fees.py",
            "ops/qualify_corporate_custody_fees.py",
            "ops/accept_historical_cost_coverage.py",
            "ops/archive_historical_cost_coverage.py",
        ]
        for path in sorted(set(paths)):
            add("project/" + path, (PROJECT / path).read_bytes())
        add(
            "implementation.patch",
            subprocess.check_output(["git", "diff", "HEAD^", "HEAD"], cwd=PROJECT),
        )
        for path in sorted(root.rglob("*")):
            if path.is_file():
                add("evidence/" + path.relative_to(root).as_posix(), path.read_bytes())
        add("dependencies.json", json.dumps(dependencies, indent=2).encode())
        z.writestr("aliases.json", json.dumps(aliases, indent=2))
        z.writestr("members.json", json.dumps(members, indent=2))
    with zipfile.ZipFile(archive) as z:
        for name, record in members.items():
            data = z.read(aliases.get(name, name))
            assert hashlib.sha256(data).hexdigest() == record["sha256"]
            assert len(data) == record["bytes"]
            target = restored / name
            assert target.resolve().is_relative_to(restored.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            assert hashlib.sha256(target.read_bytes()).hexdigest() == record["sha256"]
    books = cells = 0
    for folder in ("books", "corporate_books"):
        audit = json.loads(
            (restored / "evidence" / folder / "manifest.json").read_text(
                encoding="utf8"
            )
        )
        cases_path = Path(audit["completed"]["path"]).relative_to(root)
        cases = json.loads(
            (restored / "evidence" / cases_path).read_text(encoding="utf8")
        )
        for case in cases:
            book = restored / "evidence" / folder / case["book"]
            with np.load(book / "account.npz") as arrays:
                assert arrays["signed_shares"].shape == (case["sessions"], 933)
                cells += sum(arrays[k].size for k in arrays.files)
            meta = json.loads((book / "book.json").read_text(encoding="utf8"))
            for name, digest in meta["files"].items():
                assert hashlib.sha256((book / name).read_bytes()).hexdigest() == digest
            books += 1
    assert books == accepted["books"] and cells == accepted["account_array_cells"]
    sources = json.loads(
        (restored / "evidence/qualification.json").read_text(encoding="utf8")
    )
    source_records = [v["original"] for v in sources["sources"].values()]
    calendar = json.loads(
        (restored / "evidence/calendar_resolution.json").read_text(encoding="utf8")
    )
    source_records.append(calendar["migration_original"])
    monthly = [v["archive"] for v in sources["monthly_sources"]]
    for record in source_records + monthly:
        path = restored / "evidence" / Path(record["path"]).relative_to(root)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    assert len(source_records) == 10 and len(monthly) == 50
    report = dict(
        status="verified",
        implementation_commit=commit,
        archive={**binding(archive), "bytes": archive.stat().st_size},
        restored_root=str(restored),
        unique_members=len(unique),
        logical_members=len(members),
        aliases=len(aliases),
        restored_books=books,
        restored_account_cells=cells,
        restored_original_pdfs=len(source_records),
        restored_monthly_archives=len(monthly),
        acceptance=run["historical_cost_acceptance"],
        dependencies=dependencies,
        seconds=perf_counter() - tick,
        no_immutable_input_store_or_old_fit_copies=True,
    )
    output = base / "historical_cost_recovery.json"
    write_json_atomic(output, report)
    run["historical_cost_recovery"] = binding(output)
    write_json_atomic(pointer, run)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
