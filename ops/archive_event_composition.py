"""Recover sparse composition evidence using the immutable parent as a dependency."""

import hashlib
import json
from pathlib import Path
import re
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
    base = Path(run["root"])
    progress = bound_json(run["event_composition_progress"])
    audit = bound_json(run["event_source_composition"])
    qualification = bound_json(run["event_composition_qualification"])
    dependencies = progress["prior_recovery_dependencies"]
    old_recovery = bound_json(dependencies["spot_invoice_recovery"])
    # The frozen plan references the then-current closeout file. Preserve those
    # exact old bytes while the canonical closeout advances; never edit the plan.
    with zipfile.ZipFile(old_recovery["archive"]["path"]) as previous:
        aliases = json.loads(previous.read("aliases.json"))
        name = "project/docs/v2_STAGE_A_CLOSEOUT.md"
        old_plan = previous.read(aliases.get(name, name))
    plan = bound_json(audit["plan"])
    assert (
        hashlib.sha256(old_plan).hexdigest()
        == plan["evidence"]["stage_a_closeout_plan"]["sha256"]
    )
    snapshot = base / "event_composition/planning_closeout.md"
    snapshot.write_bytes(old_plan)
    write_json_atomic(
        base / "event_composition/planning_resolution.json",
        dict(
            original=plan["evidence"]["stage_a_closeout_plan"],
            retained=binding(snapshot),
            reason="Canonical closeout advanced after outputs; exact pre-output bytes restored from prior verified archive, frozen plan unchanged",
        ),
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    archive = base / f"event_composition_{commit[:7]}.zip"
    restored = base / f"event_composition_recovery_{commit[:7]}"
    assert not archive.exists()
    restored.mkdir(exist_ok=False)
    members, unique, aliases, rebuilt = {}, {}, {}, []
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

        paths = [
            "PROJECT_CONTEXT.md",
            "docs/v2_STAGE_A_CLOSEOUT.md",
            "docs/v2_EVENT_SOURCE_COMPOSITION.md",
            "docs/v2_economic_data_scaling_run.json",
            "docs/v2_economic_data_scaling_progress.md",
            "research/preregistrations/v2_economic_data_scaling.md",
        ]
        paths += [
            "ops/" + name
            for name in (
                "compose_stage_a_events.py",
                "qualify_event_composition.py",
                "inspect_succession_dependencies.py",
                "recover_calendar_issuer_notice.py",
                "record_event_composition.py",
                "archive_event_composition.py",
            )
        ]
        for relative in paths:
            add("project/" + relative, (PROJECT / relative).read_bytes())
        add(
            "implementation.patch",
            subprocess.check_output(["git", "diff", "HEAD^", "HEAD"], cwd=PROJECT),
        )
        for folder in ("event_composition", "calendar_issuer_notice"):
            for path in sorted((base / folder).rglob("*")):
                if not path.is_file():
                    continue
                name = "evidence/" + path.relative_to(base).as_posix()
                if path.suffix == ".npy" and re.fullmatch(
                    r"\d{4}-\d{2}-\d{2}", path.parent.name
                ):
                    rebuilt.append(
                        dict(
                            name=name,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            bytes=path.stat().st_size,
                        )
                    )
                    continue
                add(name, path.read_bytes())
        add("dependencies.json", json.dumps(dependencies, indent=2).encode())
        z.writestr("aliases.json", json.dumps(aliases, indent=2))
        z.writestr("members.json", json.dumps(members, indent=2))
        z.writestr("reconstruct_from_parent.json", json.dumps(rebuilt, indent=2))
    with zipfile.ZipFile(archive) as z:
        for name, record in members.items():
            data = z.read(aliases.get(name, name))
            assert (
                hashlib.sha256(data).hexdigest() == record["sha256"]
                and len(data) == record["bytes"]
            )
            path = restored / name
            assert path.resolve().is_relative_to(restored.resolve())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    parent = Path(audit["parent"]["root"])
    m = bound_json(
        dict(
            path=str(parent / "manifest.json"),
            sha256=audit["parent"]["manifest_sha256"],
        )
    )
    dates, isins = (
        np.load(parent / "date_index.npy"),
        np.load(parent / "isin_index.npy"),
    )
    output = restored / "evidence/event_composition"
    with np.load(output / "deltas.npz") as z:
        patches = {
            k.removesuffix("__indices"): (
                z[k].copy(),
                z[k.removesuffix("__indices") + "__values"].copy(),
            )
            for k in z.files
            if k.endswith("__indices")
        }
    cells = 0
    for record in rebuilt:
        path = restored / record["name"]
        day = int(np.searchsorted(dates, np.datetime64(path.parent.name)))
        rows = np.arange(day - 59, min(day + 11, len(dates)))
        key = path.stem
        if key == "date_index":
            value = dates[rows]
        elif key == "isin_index":
            value = isins
        else:
            array = np.load(parent / m["arrays"][key]["path"], mmap_mode="r")
            value = array[rows].copy()
            if key in patches:
                ix, values = patches[key]
                keep = (ix[:, 0] >= rows[0]) & (ix[:, 0] <= rows[-1])
                local = ix[keep].copy()
                local[:, 0] -= rows[0]
                value[tuple(local.T)] = values[keep]
        np.save(path, value)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], (
            record["name"]
        )
        cells += value.size
    report = dict(
        status="verified_partial_composition_recovery_not_stage_a_acceptance",
        implementation_commit=commit,
        archive={**binding(archive), "bytes": archive.stat().st_size},
        restored_root=str(restored),
        unique_members=len(unique),
        logical_members=len(members),
        aliases=len(aliases),
        reconstructed_view_arrays=len(rebuilt),
        reconstructed_view_cells=cells,
        target_consumer_views=qualification["samples"],
        original_pdfs=1,
        progress=run["event_composition_progress"],
        dependencies=dependencies,
        research_runtime="Unchanged research implementation in prior cost-closeout archive; exact current registration/ops/docs and implementation patch included here",
        no_immutable_input_store_or_old_fit_copies=True,
        seconds=perf_counter() - tick,
    )
    output = base / "event_composition_recovery.json"
    write_json_atomic(output, report)
    run["event_composition_recovery"] = binding(output)
    write_json_atomic(pointer, run)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
