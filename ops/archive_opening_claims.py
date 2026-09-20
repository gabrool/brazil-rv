"""Recover new claim evidence and current code without immutable input copies."""

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
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    bound_json(run["opening_claim_acceptance"])
    audit = bound_json(run["opening_claim_audit"])
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    base = Path(run["root"])
    archive = base / f"opening_claims_{commit[:7]}.zip"
    restored = base / f"opening_claims_recovery_{commit[:7]}"
    assert not archive.exists()
    restored.mkdir(exist_ok=False)
    members, unique, aliases = {}, {}, {}
    dependencies = {
        k: run[k]
        for k in (
            "held_event_source_recovery",
            "loan_return_notice_recovery",
            "economic_store_recovery",
            "corporate_replay",
            "loan_source_panels",
            "cash_calendar",
            "bova_loan_reference_audit",
            "lending_feature_recovery",
        )
    }
    for rec in dependencies.values():
        bound_json(rec)
    evidence = base / "opening_claims"
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
            "docs/v2_OPENING_CLAIM_VALUATION.md",
            "docs/v2_opening_claim_acceptance.json",
            "docs/v2_economic_data_scaling_run.json",
            "docs/v2_economic_data_scaling_progress.md",
            "ops/verify_corporate_replay.py",
        ]
        paths += [
            p.relative_to(PROJECT).as_posix()
            for p in (PROJECT / "ops").glob("*opening_claims.py")
        ]
        for name in sorted(set(paths)):
            add("project/" + name, (PROJECT / name).read_bytes())
        add(
            "implementation.patch",
            subprocess.check_output(["git", "diff", "HEAD^", "HEAD"], cwd=PROJECT),
        )
        for path in sorted(evidence.rglob("*")):
            if path.is_file():
                add(
                    "evidence/" + path.relative_to(evidence).as_posix(),
                    path.read_bytes(),
                )
        add("dependencies.json", json.dumps(dependencies, indent=2).encode())
        z.writestr("aliases.json", json.dumps(aliases, indent=2))
        z.writestr("members.json", json.dumps(members, indent=2))
    with zipfile.ZipFile(archive) as z:
        for name, record in members.items():
            data = z.read(aliases.get(name, name))
            assert (
                hashlib.sha256(data).hexdigest() == record["sha256"]
                and len(data) == record["bytes"]
            )
            target = restored / name
            assert target.resolve().is_relative_to(restored.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    cells = 0
    cases = bound_json(audit["completed"])
    for case in cases:
        path = restored / "evidence/qualified" / case["book"] / "account.npz"
        with np.load(path) as a:
            assert a["signed_shares"].shape == (30, 933)
            cells += sum(a[k].size for k in a.files)
    result = dict(
        status="verified",
        implementation_commit=commit,
        archive={**binding(archive), "bytes": archive.stat().st_size},
        unique_members=len(unique),
        logical_members=len(members),
        deduplicated_aliases=len(aliases),
        restored_root=str(restored),
        restored_books=len(cases),
        restored_account_cells=cells,
        acceptance=run["opening_claim_acceptance"],
        dependencies=dependencies,
        no_immutable_inputs_stores_or_old_fits_duplicated=True,
        seconds=perf_counter() - started,
    )
    output = base / "opening_claim_recovery.json"
    write_json_atomic(output, result)
    run["opening_claim_recovery"] = binding(output)
    write_json_atomic(pointer, run)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
