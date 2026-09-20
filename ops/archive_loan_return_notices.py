"""Archive new lifecycle evidence/code and verify restored completed account books."""

import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter
import zipfile

import numpy as np

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    pointer_path = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer_path.read_text())
    acceptance = bound_json(run["loan_return_notice_acceptance"])
    audit = bound_json(run["loan_return_notice_audit"])
    evidence = Path(run["loan_return_notice_audit"]["path"]).parent.parent
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    prior = subprocess.check_output(
        ["git", "rev-parse", "HEAD^"], cwd=PROJECT, text=True
    ).strip()
    root = Path(run["root"])
    archive = root / f"loan_return_notices_{commit[:7]}.zip"
    restore = root / f"loan_return_notices_recovery_{commit[:7]}"
    if archive.exists():
        raise FileExistsError(archive)
    restore.mkdir(exist_ok=False)
    members, unique, aliases = {}, {}, {}
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:

        def add(name, data):
            if name in members:
                raise ValueError("Duplicate member: " + name)
            digest = hashlib.sha256(data).hexdigest()
            members[name] = {"sha256": digest, "bytes": len(data)}
            if digest in unique:
                aliases[name] = unique[digest]
            else:
                z.writestr(name, data)
                unique[digest] = name

        paths = subprocess.check_output(
            ["git", "ls-files", "research"], cwd=PROJECT, text=True
        ).splitlines()
        paths += ["PROJECT_CONTEXT.md"]
        paths += [
            "docs/" + name
            for name in (
                "v2_LOAN_RETURN_NOTICES.md",
                "v2_loan_return_notice_acceptance.json",
                "v2_economic_data_scaling_run.json",
                "v2_economic_data_scaling_progress.md",
            )
        ]
        paths += [
            "ops/" + name
            for name in (
                "verify_loan_return_notices.py",
                "qualify_loan_return_notices.py",
                "seal_loan_return_notices.py",
                "archive_loan_return_notices.py",
                "verify_corporate_replay.py",
            )
        ]
        for name in sorted(set(paths)):
            add("project/" + name, (PROJECT / name).read_bytes())
        add(
            "implementation.patch",
            subprocess.check_output(["git", "diff", prior, commit], cwd=PROJECT),
        )
        for path in sorted(evidence.rglob("*")):
            if path.is_file():
                add(
                    "evidence/" + path.relative_to(evidence).as_posix(),
                    path.read_bytes(),
                )
        # Early attempts predate their explicit allocation snapshot. The old
        # implementation is in Git and unchanged during those early attempts.
        original = "research/src/brazil_rv/execution/allocation.py"
        add(
            "historical/allocation_before_polishing.py",
            subprocess.check_output(
                ["git", "show", prior + ":" + original], cwd=PROJECT
            ),
        )
        dependencies = {
            k: run[k]
            for k in (
                "economic_store_recovery",
                "lending_feature_recovery",
                "corporate_replay",
                "loan_source_panels",
                "cash_calendar",
                "bova_loan_reference_audit",
            )
        }
        for rec in dependencies.values():
            bound_json(rec)  # Manifest receipts only; no repeat source hash census.
        recipe = {
            "implementation_commit": commit,
            "prior_commit": prior,
            "acceptance": run["loan_return_notice_acceptance"],
            "source_bindings": audit["plan"]["source_bindings"],
            "dependencies": dependencies,
            "aliases": aliases,
            "historical_resolution": "Early initial/qualified/isolate allocation uses prior_commit allocation_before_polishing.py; later attempts bind executed_allocation.py. Exact other executed accounting sources and recipes are retained in each attempt where recorded. Current research plus the implementation patch reconstructs the prior source tree.",
            "restore": "Extract unique members, materialize aliases, verify members.json hashes. Completed books are evidence/adaptive_bound/<case>/account.npz. Dependencies supply unchanged sources/old store; no raw input or old fit is copied.",
        }
        # Freeze the alias map before the recipe itself is added.
        recipe["aliases"] = aliases.copy()
        add("recovery_recipe.json", (json.dumps(recipe, indent=2) + "\n").encode())
        z.writestr("members.json", json.dumps(members, indent=2))
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            target = (restore / entry.filename).resolve()
            if not target.is_relative_to(restore.resolve()):
                raise ValueError("Unsafe recovery target")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(entry))
    for name, source in recipe["aliases"].items():
        target = restore / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((restore / source).read_bytes())
    for name, rec in members.items():
        assert sha256_file(restore / name) == rec["sha256"], name
        assert (restore / name).stat().st_size == rec["bytes"], name
    cells = 0
    for case in audit["cases"]:
        with np.load(
            restore / "evidence/adaptive_bound" / case["case"] / "account.npz"
        ) as book:
            assert "loan_overdue_principal" in book.files
            assert book["signed_shares"].shape == (128, 933)
            assert np.isfinite(book["loan_overdue_principal"]).all()
            cells += sum(book[key].size for key in book.files)
    result = {
        "status": "verified",
        "implementation_commit": commit,
        "archive": {**binding(archive), "bytes": archive.stat().st_size},
        "unique_members": len(unique),
        "logical_members": len(members),
        "deduplicated_aliases": len(aliases),
        "member_index": binding(restore / "members.json"),
        "recipe": binding(restore / "recovery_recipe.json"),
        "acceptance": run["loan_return_notice_acceptance"],
        "restored_complete_books": len(audit["cases"]),
        "restored_account_array_cells": cells,
        "dependencies": dependencies,
        "no_raw_inputs_or_old_fits_duplicated": True,
        "stage_status": acceptance["status"],
        "seconds": perf_counter() - started,
    }
    receipt = root / "loan_return_notice_recovery.json"
    write_json_atomic(receipt, result)
    run["loan_return_notice_recovery"] = binding(receipt)
    write_json_atomic(pointer_path, run)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
