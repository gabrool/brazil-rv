"""Recover new invoice books and bounded calendar evidence without old-data copies."""

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
    accepted = bound_json(run["spot_invoice_acceptance"])
    base = Path(run["root"])
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    archive = base / f"cost_closeout_{commit[:7]}.zip"
    restored = base / f"cost_closeout_recovery_{commit[:7]}"
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
            "docs/v2_SPOT_INVOICE_CLOSEOUT.md",
            "docs/v2_spot_invoice_acceptance.json",
            "docs/v2_economic_data_scaling_run.json",
            "docs/v2_economic_data_scaling_progress.md",
            "ops/verify_corporate_replay.py",
            "ops/audit_spot_invoices.py",
            "ops/qualify_spot_invoices.py",
            "ops/recover_closeout_calendar.py",
            "ops/accept_spot_invoice_closeout.py",
            "ops/archive_spot_invoice_closeout.py",
        ]
        for path in sorted(set(paths)):
            add("project/" + path, (PROJECT / path).read_bytes())
        add(
            "implementation.patch",
            subprocess.check_output(["git", "diff", "HEAD^", "HEAD"], cwd=PROJECT),
        )
        for folder in ("spot_invoice", "closeout_calendar_sources"):
            for path in sorted((base / folder).rglob("*")):
                if path.is_file():
                    add(
                        "evidence/" + path.relative_to(base).as_posix(),
                        path.read_bytes(),
                    )
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
    root = restored / "evidence/spot_invoice"
    cases = json.loads((root / "completed.json").read_text(encoding="utf8"))
    cells = 0
    for case in cases:
        book = root / case["book"]
        with np.load(book / "account.npz") as arrays:
            assert arrays["signed_shares"].shape == (case["sessions"], 933)
            cells += sum(arrays[k].size for k in arrays.files)
        meta = json.loads((book / "book.json").read_text(encoding="utf8"))
        for name, digest in meta["files"].items():
            assert hashlib.sha256((book / name).read_bytes()).hexdigest() == digest
    assert len(cases) == accepted["books"] and cells == accepted["account_array_cells"]
    sources = json.loads(
        (restored / "evidence/closeout_calendar_sources/qualification.json").read_text(
            encoding="utf8"
        )
    )
    for record in sources["originals"].values():
        path = restored / "evidence" / Path(record["path"]).relative_to(base)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    report = dict(
        status="verified",
        implementation_commit=commit,
        archive={**binding(archive), "bytes": archive.stat().st_size},
        restored_root=str(restored),
        unique_members=len(unique),
        logical_members=len(members),
        aliases=len(aliases),
        restored_books=len(cases),
        restored_account_cells=cells,
        restored_calendar_pdf_leads=len(sources["originals"]),
        calendar_scope="One dated calendar original, one2011third-party mirror and one explicitly unapproved2017B3draft. No new calendar resolution or bound.",
        acceptance=run["spot_invoice_acceptance"],
        dependencies=dependencies,
        seconds=perf_counter() - tick,
        no_immutable_input_store_or_old_fit_copies=True,
    )
    output = base / "spot_invoice_recovery.json"
    write_json_atomic(output, report)
    run["spot_invoice_recovery"] = binding(output)
    write_json_atomic(pointer, run)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
