"""Restore and hash-check only new surviving-issuer dependency evidence."""

import hashlib
import io
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
    run = json.loads(pointer.read_text())
    progress = bound_json(run["surviving_rename_dependency_progress"])
    source = Path(run["surviving_rename_admission"]["path"]).parent
    parent = Path(progress["parent"]["root"])
    bound_json(
        dict(
            path=str(parent / "manifest.json"),
            sha256=progress["parent"]["manifest_sha256"],
        )
    )
    slow = bound_json(run["surviving_rename_daily_input_audit"])["view"]
    bound_json(slow)
    slow_root = Path(slow["path"]).parent
    offset = int(
        np.searchsorted(
            np.load(parent / "date_index.npy"), np.load(slow_root / "date_index.npy")[0]
        )
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    archive = source.parent / f"surviving_dependencies_{commit[:7]}.zip"
    restore = source.parent / f"surviving_dependencies_recovery_{commit[:7]}"
    assert not archive.exists()
    restore.mkdir(exist_ok=False)
    members, unique, aliases, arrays = {}, {}, {}, []

    def baseline(record):
        root = slow_root if record["base"] == "prior_slow_view" else parent
        value = np.load(root / (record["key"] + ".npy"), mmap_mode="r")
        return np.array(value[record["first"] : record["stop"]], copy=True)

    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:

        def add(name, data):
            digest = hashlib.sha256(data).hexdigest()
            assert name not in members
            members[name] = dict(sha256=digest, bytes=len(data))
            if digest in unique:
                aliases[name] = unique[digest]
            else:
                unique[digest] = name
                z.writestr(name, data)

        for relative in [
            "PROJECT_CONTEXT.md",
            "docs/v2_STAGE_A_CLOSEOUT.md",
            "docs/v2_SURVIVING_RENAME_PROPAGATION.md",
            "docs/v2_economic_data_scaling_progress.md",
            "docs/v2_economic_data_scaling_run.json",
            "research/preregistrations/v2_economic_data_scaling.md",
            *[
                "ops/" + n
                for n in (
                    "propagate_surviving_issuers.py",
                    "propagate_surviving_context.py",
                    "propagate_surviving_lending.py",
                    "qualify_surviving_unit_history.py",
                    "verify_surviving_dependencies.py",
                    "propagate_surviving_auxiliaries.py",
                    "qualify_surviving_auxiliaries.py",
                    "record_surviving_dependencies.py",
                    "archive_surviving_dependencies.py",
                    "propagate_remaining_auxiliaries.py",
                    "propagate_fca_peers.py",
                    "propagate_fca_financials.py",
                    "propagate_rename_issuers.py",
                    "audit_auxiliary_tensors.py",
                )
            ],
        ]:
            add("project/" + relative, (PROJECT / relative).read_bytes())
        add(
            "implementation.patch",
            subprocess.check_output(["git", "diff", "HEAD^", "HEAD"], cwd=PROJECT),
        )
        for folder in (
            "issuers",
            "context",
            "lending",
            "unit_history",
            "dependency_consumer",
            "auxiliaries",
            "auxiliary_consumer",
            "dependency_progress",
        ):
            for path in sorted((source / folder).rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(source).as_posix()
                name = "evidence/" + relative
                if path.suffix != ".npy" or path.parent.name != "view":
                    add(name, path.read_bytes())
                    continue
                value = np.load(path, mmap_mode="r")
                key = path.stem
                reused = key in (
                    "active",
                    "slow_values",
                    "slow_valid",
                    "slow_age_sessions",
                    "slow_timestep_valid",
                    "date_index",
                    "isin_index",
                )
                first = 0 if reused else offset
                record = dict(
                    name=name,
                    key=key,
                    base="prior_slow_view" if reused else "accepted_natura_parent",
                    first=first,
                    stop=first + len(value),
                    sha256=binding(path)["sha256"],
                    bytes=path.stat().st_size,
                )
                initial = baseline(record)
                assert initial.shape == value.shape and initial.dtype == value.dtype
                ix = np.argwhere(
                    value.view(f"V{value.dtype.itemsize}")
                    != initial.view(f"V{initial.dtype.itemsize}")
                )
                buffer = io.BytesIO()
                np.savez_compressed(buffer, indices=ix, values=value[tuple(ix.T)])
                record["delta"] = "sparse/" + relative + ".npz"
                record["changed_cells"] = len(ix)
                add(record["delta"], buffer.getvalue())
                arrays.append(record)
        for name in (
            "surviving_issuers_stdout.log",
            "surviving_context_stdout.log",
            "surviving_context_resume_stdout.log",
            "surviving_lending_stdout.log",
            "surviving_unit_history_stdout.log",
            "surviving_dependency_consumer_stdout.log",
            "surviving_auxiliaries_stdout.log",
            "surviving_auxiliary_consumer_stdout.log",
            "surviving_auxiliary_arithmetic_stdout.log",
            "surviving_auxiliary_arithmetic_resume_stdout.log",
            "surviving_dependency_progress_stdout.log",
        ):
            add("stdout/" + name, (source.parent / name).read_bytes())
        add(
            "dependencies.json",
            json.dumps(progress["prior_recovery_dependencies"], indent=2).encode(),
        )
        z.writestr("aliases.json", json.dumps(aliases, indent=2))
        z.writestr("members.json", json.dumps(members, indent=2))
        z.writestr(
            "reconstruct.json",
            json.dumps(
                dict(parent=progress["parent"], prior_slow_view=slow, arrays=arrays),
                indent=2,
            ),
        )
    with zipfile.ZipFile(archive) as z:
        for name, rec in members.items():
            data = z.read(aliases.get(name, name))
            assert (
                len(data) == rec["bytes"]
                and hashlib.sha256(data).hexdigest() == rec["sha256"]
            )
            path = restore / name
            assert path.resolve().is_relative_to(restore.resolve())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            assert binding(path)["sha256"] == rec["sha256"]
    cells = 0
    for record in arrays:
        value = baseline(record)
        with np.load(restore / record["delta"]) as z:
            value[tuple(z["indices"].T)] = z["values"]
        path = restore / record["name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, value)
        assert binding(path)["sha256"] == record["sha256"]
        cells += value.size
    report = dict(
        status="verified_sparse_dependency_recovery",
        implementation_commit=commit,
        archive={**binding(archive), "bytes": archive.stat().st_size},
        unique_members=len(unique),
        logical_members=len(members),
        aliases=len(aliases),
        reconstructed_arrays=len(arrays),
        reconstructed_cells=cells,
        changed_arrays=sum(r["changed_cells"] > 0 for r in arrays),
        prior_view_arrays_reused=sum(r["base"] == "prior_slow_view" for r in arrays),
        parent=progress["parent"],
        prior_slow_view=slow,
        dependencies=progress["prior_recovery_dependencies"],
        immutable_inputs_copied=False,
        limits="Current new ops/helpers/docs/patch, exact failed/resumed/executed recipes and new evidence recover. Unchanged research runtime and original sources remain prior archive dependencies. Array counts include overlapping audit views; recovery is byte verification, not another daily/native/model audit.",
        seconds=perf_counter() - tick,
    )
    output = source.parent / "surviving_rename_dependency_recovery.json"
    write_json_atomic(output, report)
    write_json_atomic(PROJECT / "docs/v2_surviving_dependency_recovery.json", report)
    run["surviving_rename_dependency_recovery"] = binding(output)
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "archive",
                    "unique_members",
                    "logical_members",
                    "reconstructed_arrays",
                    "reconstructed_cells",
                    "changed_arrays",
                    "seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
