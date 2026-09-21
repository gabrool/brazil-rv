"""Recover the completed expansion, without copying old fits or source stores."""

from copy import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
from time import perf_counter
import zipfile

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.corporate_replay import apply_corporate_replay, load_corporate_replay
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    fits = bound_json(run["scaling_expanded_fits"])
    assert fits["status"] == "complete" and len(fits["completed"]) == 54
    bound_json(run["scaling_expanded_results"])
    assert bound_json(run["scaling_hedge_roundoff_account_parity"])["passed"]
    root = Path(run["scaling_expanded_fit_plan"]["path"]).parent.resolve()
    hedge = Path(run["scaling_hedge_roundoff_plan"]["path"]).parent.resolve()
    prior = bound_json(run["scaling_diagnostic_recovery"])
    with zipfile.ZipFile(prior["archive"]["path"]) as source:
        known = json.loads(source.read("inherited_members.json"))
        known.update(json.loads(source.read("members.json")))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    # C provides staging space; the verified final archive lives on D. Verified
    # redundant new-fit epoch copies can then release enough D space to move it.
    staging = Path(run["stage_c_root"]) / f"attention_expansion_recovery_{commit[:7]}"
    staging.mkdir(exist_ok=False)
    archive = staging / f"attention_expansion_{commit[:7]}.zip"
    final = Path(run["root"]).resolve() / archive.name
    assert not final.exists() and final.parent == Path(run["root"]).resolve()
    source_inputs = [
        bound_json(bound_json(run[key])["inputs"])
        for key in (
            "scaling_expanded_source_plan",
            "scaling_expanded_later_source_plan",
        )
    ]
    caches = {Path(inputs["cache"]["path"]).resolve() for inputs in source_inputs}
    members, inherited, aliases, unique, retired = {}, {}, {}, {}, []
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as output:

        def add(name, path):
            receipt = dict(sha256=sha256_file(path), bytes=path.stat().st_size)
            if known.get(name) == receipt:
                inherited[name] = receipt
                return
            assert name not in members
            members[name] = receipt
            if receipt["sha256"] in unique:
                aliases[name] = unique[receipt["sha256"]]
            else:
                unique[receipt["sha256"]] = name
                output.write(path, name)

        tracked = subprocess.check_output(
            ["git", "ls-files", "research", "ops", "docs", "PROJECT_CONTEXT.md"],
            cwd=PROJECT,
            text=True,
        ).splitlines()
        for name in tracked:
            add("project/" + name, PROJECT / name)
        for folder, label in ((root, "expanded"), (hedge, "hedge_roundoff")):
            for directory, children, files in os.walk(folder):
                children[:] = [
                    n
                    for n in children
                    if n != "__pycache__" and not (Path(directory) / n).is_junction()
                ]
                for name in sorted(files):
                    path = Path(directory) / name
                    assert path.resolve().is_relative_to(folder)
                    if path.resolve() in caches:
                        continue
                    member = (
                        "evidence/" + label + "/" + path.relative_to(folder).as_posix()
                    )
                    add(member, path)
                    if (
                        folder == root
                        and path.parent.name == "epochs"
                        and re.fullmatch(r"epoch_\d+\.pt", name)
                    ):
                        assert path.parent.parent.name.split("_")[0] in {
                            "F3",
                            "F7",
                            "F11",
                            "F13",
                        }
                        assert path.resolve().is_relative_to(root / "fits")
                        assert (path.parent.parent / "selected.pt").exists()
                        assert (path.parent.parent / "selected_ema.pt").exists()
                        retired.append(
                            dict(path=str(path), member=member, **members[member])
                        )
        for name, value in (
            ("members", members),
            ("inherited_members", inherited),
            ("aliases", aliases),
            ("redundant_epochs", retired),
        ):
            output.writestr(name + ".json", json.dumps(value, indent=2))
        output.writestr(
            "dependencies.json",
            json.dumps(
                dict(
                    prior=run["scaling_diagnostic_recovery"],
                    source_inputs=source_inputs,
                ),
                indent=2,
            ),
        )
    archive_seconds = perf_counter() - started
    target = staging / "one_file.bin"
    with zipfile.ZipFile(archive) as source:
        for name, receipt in members.items():
            with (
                source.open(aliases.get(name, name)) as compressed,
                target.open("wb") as restored,
            ):
                shutil.copyfileobj(compressed, restored, 1024 * 1024)
            assert (
                target.stat().st_size == receipt["bytes"]
                and sha256_file(target) == receipt["sha256"]
            ), name
            target.unlink()
        recovered_caches = []
        for inputs in source_inputs:
            term_name = (
                "evidence/expanded/"
                + Path(inputs["account_terms"]["path"])
                .resolve()
                .relative_to(root)
                .as_posix()
            )
            terms = staging / "incremental_terms.json"
            terms.write_bytes(source.read(aliases.get(term_name, term_name)))
            # Stream reconstructed pickle bytes into a digest, avoiding another
            # 707MB temporary cache on the nearly full staging volume.
            parent = bound_json(inputs["parent"])["cache"]
            with Path(parent["path"]).open("rb") as f:
                data = pickle.load(f)
            provenance = dict(data.inputs.source_artifact_hashes)
            provenance["corporate_replay_parent"] = provenance.pop("corporate_replay")
            loaded, calendar = load_corporate_replay(
                str(terms), inputs["account_terms"]["sha256"]
            )
            amended = copy(data)
            amended.inputs = apply_corporate_replay(
                replace(data.inputs, source_artifact_hashes=provenance),
                loaded,
                calendar,
                inputs["account_terms"]["sha256"],
            )
            digest = hashlib.sha256()
            byte_count = 0

            class DigestWriter:
                def write(self, chunk):
                    nonlocal byte_count
                    digest.update(chunk)
                    byte_count += len(chunk)
                    return len(chunk)

            pickle.dump(amended, DigestWriter(), protocol=pickle.HIGHEST_PROTOCOL)
            assert digest.hexdigest() == inputs["cache"]["sha256"]
            recovered_caches.append(dict(cache=inputs["cache"], bytes=byte_count))
            terms.unlink()
            del data, amended
    verified_hash = sha256_file(archive)
    # Only exact new-fit intermediate files verified above are retired. This is
    # not recursive deletion and cannot traverse old-fit junctions.
    for row in retired:
        path = Path(row["path"])
        assert path.resolve().is_relative_to(root / "fits")
        assert path.parent.name == "epochs" and sha256_file(path) == row["sha256"]
        path.unlink()
    assert shutil.disk_usage(final.parent).free > archive.stat().st_size
    shutil.copyfile(archive, final)
    assert sha256_file(final) == verified_hash
    archive.unlink()
    staging.rmdir()
    report = dict(
        status="verified_expanded_attention_recovery",
        implementation_commit=commit,
        archive={**binding(final), "bytes": final.stat().st_size},
        logical_members=len(members),
        unique_members=len(unique),
        aliases=len(aliases),
        inherited_members=len(inherited),
        recovered_caches=recovered_caches,
        recovered_cache_bytes=sum(r["bytes"] for r in recovered_caches),
        retired_epochs=retired,
        retired_logical_bytes=sum(r["bytes"] for r in retired),
        archive_seconds=archive_seconds,
        recovery_seconds=perf_counter() - started - archive_seconds,
        dependencies=dict(
            prior=run["scaling_diagnostic_recovery"],
            source_parents=[inputs["parent"] for inputs in source_inputs],
        ),
        scope="All24 new completed child fits and96 baseline book records, source overlays/bounds, hedge correction, selected originals/leads, exact attempts and current research restore/hash-check. Thirty old-fit junctions remain dependencies. Both omitted shallow account caches reconstruct as byte-exact pickle streams from their prior immutable cache and restored terms; streams are hashed without writing redundant cache files. Every removed new intermediate epoch is physically restored and verified in this archive; selected/EMA weights, histories, forecasts and all original fits remain online. No raw or accepted-store copies, source census or repeated numerical proof.",
    )
    destination = Path(run["root"]) / "attention_expansion_recovery.json"
    write_json_atomic(destination, report)
    write_json_atomic(PROJECT / "docs/v2_attention_expansion_recovery.json", report)
    run = json.loads(pointer.read_text())
    run["scaling_expanded_recovery"] = binding(destination)
    write_json_atomic(pointer, run)
    print(
        json.dumps({k: v for k, v in report.items() if k != "retired_epochs"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
