"""Recover the integrated account admission without immutable input duplication."""

import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter
import zipfile


from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    bound_json(run["economic_account_acceptance"])
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    base = Path(run["root"])
    archive = base / f"integrated_account_{commit[:7]}.zip"
    restored = base / f"integrated_account_recovery_{commit[:7]}"
    assert not archive.exists()
    restored.mkdir(exist_ok=False)
    dependencies = {
        k: run[k]
        for k in (
            "january_calendar_recovery",
            "historical_cost_recovery",
            "spot_invoice_recovery",
            "composed_store_recovery",
            "event_composition_recovery",
            "loan_return_notice_recovery",
            "lending_feature_recovery",
            "cash_calendar",
            "loan_source_panels",
            "bova_loan_reference_audit",
            "enat_settlement_terms",
        )
    }
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
            "docs/v2_JANUARY_CALENDAR_HYPOTHESES.md",
            "docs/v2_JANUARY_CALENDAR_ACCEPTANCE.md",
            "docs/v2_economic_account_acceptance.json",
            "docs/v2_economic_data_scaling_run.json",
            "docs/v2_economic_data_scaling_progress.md",
            "docs/v2_STAGE_A_CLOSEOUT.md",
            "ops/verify_corporate_replay.py",
            "ops/accept_economic_account.py",
            "ops/inspect_surviving_account_exposure.py",
            "ops/archive_integrated_account.py",
        ]
        for path in sorted(set(paths)):
            add("project/" + path, (PROJECT / path).read_bytes())
        add(
            "implementation.patch",
            subprocess.check_output(["git", "diff", "HEAD^", "HEAD"], cwd=PROJECT),
        )
        for folder in ("integrated_admission",):
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
            assert (
                hashlib.sha256(data).hexdigest() == record["sha256"]
                and len(data) == record["bytes"]
            )
            target = restored / name
            assert target.resolve().is_relative_to(restored.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            assert hashlib.sha256(target.read_bytes()).hexdigest() == record["sha256"]
    report = dict(
        status="verified",
        implementation_commit=commit,
        archive={**binding(archive), "bytes": archive.stat().st_size},
        restored_root=str(restored),
        unique_members=len(unique),
        logical_members=len(members),
        aliases=len(aliases),
        new_historical_books=0,
        acceptance=run["economic_account_acceptance"],
        dependencies=dependencies,
        seconds=perf_counter() - tick,
        no_immutable_input_store_or_old_fit_copies=True,
    )
    output = base / "integrated_account_recovery.json"
    write_json_atomic(output, report)
    run["integrated_account_recovery"] = binding(output)
    write_json_atomic(pointer, run)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
