"""Recover new originals, exact recipes and sparse Natura amendments on D."""

import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter
import zipfile

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.corporate_actions import verified_action_terms_from_table
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    acceptance = bound_json(run["held_event_source_acceptance"])
    reports = {key: bound_json(rec) for key, rec in acceptance["evidence"].items()}
    root = Path(run["root"])
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    prior = subprocess.check_output(
        ["git", "rev-parse", "HEAD^"], cwd=PROJECT, text=True
    ).strip()
    archive = root / f"held_event_sources_{commit[:7]}.zip"
    restore = root / f"held_event_sources_recovery_{commit[:7]}"
    if archive.exists():
        raise FileExistsError(archive)
    restore.mkdir(exist_ok=False)
    members, unique, aliases = {}, {}, {}
    dependencies = {
        k: run[k]
        for k in (
            "economic_store_recovery",
            "loan_return_notice_recovery",
            "bvbg_target_recovery",
            "corporate_replay",
        )
    }
    for rec in dependencies.values():
        bound_json(rec)
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:

        def add(name, data):
            if name in members:
                raise ValueError("Duplicate member " + name)
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
        paths += [
            "PROJECT_CONTEXT.md",
            "docs/v2_HELD_EVENT_SOURCE_AUDIT.md",
            "docs/v2_held_event_source_acceptance.json",
            "docs/v2_economic_data_scaling_run.json",
            "docs/v2_economic_data_scaling_progress.md",
        ]
        paths += [
            "ops/" + s + ".py"
            for s in (
                "recover_held_event_sources",
                "audit_held_event_sources",
                "qualify_held_event_sources",
                "propagate_natura_gross_targets",
                "propagate_natura_wealth",
                "seal_held_event_sources",
                "archive_held_event_sources",
                "propagate_corporate_targets",
            )
        ]
        for name in sorted(set(paths)):
            add("project/" + name, (PROJECT / name).read_bytes())
        add(
            "implementation.patch",
            subprocess.check_output(["git", "diff", prior, commit], cwd=PROJECT),
        )
        for family in ("held_event_sources", "natura_gross_targets", "natura_wealth"):
            for path in sorted((root / family).rglob("*")):
                if path.is_file():
                    add(
                        "evidence/" + path.relative_to(root).as_posix(),
                        path.read_bytes(),
                    )
        recipe = {
            "implementation_commit": commit,
            "prior_commit": prior,
            "acceptance": run["held_event_source_acceptance"],
            "dependencies": dependencies,
            "aliases": aliases.copy(),
            "restore": "Extract members, materialize aliases, verify hashes/bytes. New sources and exact initial/qualified recipes are under evidence/. The accepted store is an explicit dependency; restore target rows and wealth rows by applying the qualified sparse indices on their recorded baseline rows. Do not use these intermediate risk-fixed targets as final training inputs. No immutable annual source or old fit is duplicated.",
        }
        add("recovery_recipe.json", (json.dumps(recipe, indent=2) + "\n").encode())
        z.writestr("members.json", json.dumps(members, indent=2))
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            target = (restore / entry.filename).resolve()
            if not target.is_relative_to(restore.resolve()):
                raise ValueError("Unsafe recovery target")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(entry))
    for name, original in recipe["aliases"].items():
        target = restore / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((restore / original).read_bytes())
    for name, rec in members.items():
        assert sha256_file(restore / name) == rec["sha256"]
        assert (restore / name).stat().st_size == rec["bytes"]

    def recovered(path):
        return restore / "evidence" / Path(path).relative_to(root)

    clock = reports["held_event_source_qualification"]
    terms = verified_action_terms_from_table(
        pl.read_parquet(recovered(clock["amendments"]["path"]))
    )
    assert len(terms) == 2 and [t.shares_per_prior_share for t in terms] == [2, 1]
    targets = reports["natura_gross_target_attribution"]
    source = Path(targets["parent_store"]["root"])
    manifest = bound_json(
        {
            "path": str(source / "manifest.json"),
            "sha256": targets["parent_store"]["manifest_sha256"],
        }
    )
    rows = np.load(recovered(targets["rows"]["path"]))
    target_cells = 0
    with np.load(recovered(targets["deltas"]["path"])) as z:
        for name in z.files:
            if not name.endswith("__indices"):
                continue
            key = name.removesuffix("__indices")
            result = np.load(source / manifest["arrays"][key]["path"], mmap_mode="r")[
                rows
            ].copy()
            indices = z[name].copy()
            indices[:, 0] = np.searchsorted(rows, indices[:, 0])
            result[tuple(indices.T)] = z[key + "__values"]
            np.testing.assert_array_equal(
                result,
                np.load(
                    recovered(targets["deltas"]["path"]).parent / (key + "_rows.npy")
                ),
            )
            target_cells += result.size
    wealth = reports["natura_wealth_amendment"]
    wrows = np.load(recovered(wealth["rows"]["path"]))
    values = np.load(recovered(wealth["rows"]["path"]).parent / "wealth_rows.npy")
    axis = np.load(source / "isin_index.npy").tolist().index("BRNATUACNOR6")
    with np.load(recovered(wealth["deltas"]["path"])) as z:
        for k, field in enumerate(("open", "high", "low", "close")):
            key = "shareholder_wealth_" + field
            result = np.load(source / manifest["arrays"][key]["path"], mmap_mode="r")[
                wrows, axis
            ].copy()
            indices = z[key + "__indices"]
            assert (indices[:, 1] == axis).all()
            result[np.searchsorted(wrows, indices[:, 0])] = z[key + "__values"]
            np.testing.assert_array_equal(result, values[:, k])
    result = {
        "status": "verified",
        "implementation_commit": commit,
        "archive": {**binding(archive), "bytes": archive.stat().st_size},
        "unique_members": len(unique),
        "logical_members": len(members),
        "deduplicated_aliases": len(aliases),
        "member_index": binding(restore / "members.json"),
        "recipe": binding(restore / "recovery_recipe.json"),
        "acceptance": run["held_event_source_acceptance"],
        "restored_original_pdfs": acceptance["source_counts"]["new_original_pdfs"],
        "restored_loader_terms": len(terms),
        "reconstructed_target_cells": target_cells,
        "reconstructed_wealth_cells": values.size,
        "dependencies": dependencies,
        "no_immutable_inputs_accepted_store_or_old_fits_duplicated": True,
        "seconds": perf_counter() - started,
    }
    receipt = root / "held_event_source_recovery.json"
    write_json_atomic(receipt, result)
    run["held_event_source_recovery"] = binding(receipt)
    write_json_atomic(pointer, run)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
