"""Recover new rename evidence without copying accepted store arrays."""

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
    run = json.loads(pointer.read_text(encoding="utf8"))
    progress = bound_json(run["surviving_rename_progress"])
    source = Path(run["root"]) / "surviving_renames"
    parent = Path(progress["parent"]["root"])
    m = bound_json(
        dict(
            path=str(parent / "manifest.json"),
            sha256=progress["parent"]["manifest_sha256"],
        )
    )
    plan = bound_json(binding(source / "daily/plan.json"))
    start, first, stop = plan["rows"]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    archive = source.parent / f"surviving_renames_{commit[:7]}.zip"
    restore = source.parent / f"surviving_renames_recovery_{commit[:7]}"
    assert not archive.exists()
    restore.mkdir(exist_ok=False)
    members, unique, aliases, rebuilt = {}, {}, {}, []

    def base_array(record):
        key = record["key"]
        path = parent / (
            m["arrays"][key]["path"] if key in m["arrays"] else key + ".npy"
        )
        a = np.load(path, mmap_mode="r")
        value = np.array(
            a[record["start"] : record["stop"]],
            dtype=np.dtype(record["dtype"]),
            copy=True,
        )
        return ~value if record.get("invert") else value

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

        paths = [
            "PROJECT_CONTEXT.md",
            "docs/v2_STAGE_A_CLOSEOUT.md",
            "docs/v2_SURVIVING_RENAME_PROPAGATION.md",
            "docs/v2_economic_data_scaling_run.json",
            "docs/v2_economic_data_scaling_progress.md",
            "research/preregistrations/v2_economic_data_scaling.md",
            "research/configs/v2/isin_links_allowlist.csv",
        ]
        paths.extend(
            "ops/" + n
            for n in (
                "admit_surviving_renames.py",
                "propagate_surviving_wealth.py",
                "propagate_surviving_daily.py",
                "qualify_surviving_daily.py",
                "finish_surviving_daily.py",
                "verify_surviving_daily_inputs.py",
                "record_surviving_renames.py",
                "archive_surviving_renames.py",
            )
        )
        for relative in paths:
            add("project/" + relative, (PROJECT / relative).read_bytes())
        add(
            "implementation.patch",
            subprocess.check_output(["git", "diff", "HEAD^", "HEAD"], cwd=PROJECT),
        )
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source).as_posix()
            name = "evidence/" + relative
            key = None
            offset = start
            invert = False
            if path.suffix == ".npy":
                if relative.startswith("consumer/slow_view/"):
                    key = path.stem
                    offset = first - 60
                    if key == "isin_index":
                        offset = 0
                elif path.parent.name in ("control", "corrected"):
                    key = {
                        "active": "active",
                        "activity": "activity_valid",
                        "volume": "volume_brl",
                        "ambiguous": "action_session_resolved",
                        "clusters": "monthly_cluster_labels",
                    }.get(path.stem)
                    if path.stem.startswith("wealth_"):
                        key = "shareholder_" + path.stem
                    invert = path.stem == "ambiguous"
                elif relative == "daily/date_index.npy":
                    key = "date_index"
            if key is None:
                add(name, path.read_bytes())
                continue
            value = np.load(path)
            record = dict(
                name=name,
                key=key,
                start=offset,
                stop=offset + len(value),
                dtype=value.dtype.str,
                invert=invert,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size,
            )
            baseline = base_array(record)
            assert baseline.shape == value.shape
            # Byte comparisons preserve signed zeros and every NaN payload.
            ix = np.argwhere(
                value.view(f"V{value.dtype.itemsize}")
                != baseline.view(f"V{baseline.dtype.itemsize}")
            )
            buf = io.BytesIO()
            np.savez_compressed(buf, indices=ix, values=value[tuple(ix.T)])
            record["delta"] = "sparse/" + relative + ".npz"
            add(record["delta"], buf.getvalue())
            rebuilt.append(record)
        add(
            "dependencies.json",
            json.dumps(progress["prior_recovery_dependencies"], indent=2).encode(),
        )
        z.writestr("aliases.json", json.dumps(aliases, indent=2))
        z.writestr("members.json", json.dumps(members, indent=2))
        z.writestr(
            "reconstruct.json",
            json.dumps(dict(parent=progress["parent"], arrays=rebuilt), indent=2),
        )
    with zipfile.ZipFile(archive) as z:
        for name, record in members.items():
            data = z.read(aliases.get(name, name))
            assert (
                len(data) == record["bytes"]
                and hashlib.sha256(data).hexdigest() == record["sha256"]
            )
            path = restore / name
            assert path.resolve().is_relative_to(restore.resolve())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    cells = 0
    for record in rebuilt:
        value = base_array(record)
        with np.load(restore / record["delta"]) as z:
            value[tuple(z["indices"].T)] = z["values"]
        path = restore / record["name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, value)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        cells += value.size
    report = {
        "status": "verified_sparse_recovery",
        "implementation_commit": commit,
        "archive": {**binding(archive), "bytes": archive.stat().st_size},
        "unique_members": len(unique),
        "logical_members": len(members),
        "aliases": len(aliases),
        "reconstructed_arrays": len(rebuilt),
        "reconstructed_cells": cells,
        "consumer_samples": 185,
        "parent": progress["parent"],
        "dependencies": progress["prior_recovery_dependencies"],
        "immutable_inputs_copied": False,
        "runtime": "Unchanged research runtime uses prior verified event/cost recovery; current ops/config/docs/patch and exact failed-qualified executed recipes included.",
        "seconds": perf_counter() - tick,
    }
    output = source.parent / "surviving_rename_recovery.json"
    write_json_atomic(output, report)
    run["surviving_rename_recovery"] = binding(output)
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
                    "seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
