"""Resume only cache-stream verification after the saved PickleBuffer failure."""

from copy import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import shutil
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
    root = Path(run["scaling_expanded_fit_plan"]["path"]).parent.resolve()
    epochs = bound_json(binding(root / "epoch_recovery.json"))
    commit = Path(epochs["archive"]["path"]).stem.rsplit("_", 1)[1]
    staging = Path(run["stage_c_root"]) / f"attention_expansion_recovery_{commit}"
    archive = staging / f"attention_expansion_{commit}.zip"
    failure = Path(run["stage_c_root"]) / "attention_expansion_archive_stdout.txt"
    assert (
        "TypeError: object of type 'pickle.PickleBuffer' has no len()"
        in failure.read_text()
    )
    attempts = root / "recovery_attempts"
    attempts.mkdir(exist_ok=False)
    shutil.copyfile(failure, attempts / "initial_stdout.txt")
    (attempts / "resume_executed.py").write_bytes(Path(__file__).read_bytes())
    with zipfile.ZipFile(archive) as source:
        members = json.loads(source.read("members.json"))
        aliases = json.loads(source.read("aliases.json"))
        inherited = json.loads(source.read("inherited_members.json"))
        dependencies = json.loads(source.read("dependencies.json"))
        (attempts / "initial_executed.py").write_bytes(
            source.read("project/ops/archive_attention_expansion.py")
        )
        caches = []
        for inputs in dependencies["source_inputs"]:
            name = (
                "evidence/expanded/"
                + Path(inputs["account_terms"]["path"])
                .resolve()
                .relative_to(root)
                .as_posix()
            )
            terms = staging / "incremental_terms.json"
            terms.write_bytes(source.read(aliases.get(name, name)))
            with Path(bound_json(inputs["parent"])["cache"]["path"]).open(
                "rb"
            ) as handle:
                data = pickle.load(handle)
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
            digest, count = hashlib.sha256(), 0

            class DigestWriter:
                def write(self, chunk):
                    nonlocal count
                    digest.update(chunk)
                    size = memoryview(chunk).nbytes
                    count += size
                    return size

            pickle.dump(amended, DigestWriter(), protocol=pickle.HIGHEST_PROTOCOL)
            assert digest.hexdigest() == inputs["cache"]["sha256"]
            caches.append(dict(cache=inputs["cache"], bytes=count))
            terms.unlink()
            del data, amended
    # The exact initial traceback is after the full member restore/hash loop.
    # Retain that demonstrated execution boundary rather than rerun the archive.
    disposition = dict(
        reused_member_verification=True,
        basis="Exact saved initial recipe and traceback reach cache pickle.dump only after every ordinary member restored and hash-checked. Counts are reconstructed from the saved ZIP inventory, not a separately saved successful phase receipt.",
        initial_archive=binding(archive),
        recovered_caches=caches,
        canonical_archiver=binding(PROJECT / "ops/archive_attention_expansion.py"),
    )
    write_json_atomic(attempts / "qualification.json", disposition)
    shutil.copyfile(
        PROJECT / "ops/archive_attention_expansion.py",
        attempts / "qualified_archiver.py",
    )
    extra = {}
    with zipfile.ZipFile(archive, "a", zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(attempts.iterdir()):
            name = "recovery_resume/" + path.name
            assert name not in output.namelist()
            extra[name] = dict(sha256=sha256_file(path), bytes=path.stat().st_size)
            output.write(path, name)
        output.writestr("recovery_resume/members.json", json.dumps(extra, indent=2))
    target = staging / "one_resume_file.bin"
    with zipfile.ZipFile(archive) as source:
        for name, receipt in extra.items():
            target.write_bytes(source.read(name))
            assert (
                sha256_file(target) == receipt["sha256"]
                and target.stat().st_size == receipt["bytes"]
            )
            target.unlink()
    final = Path(run["root"]).resolve() / archive.name
    assert (
        not final.exists()
        and shutil.disk_usage(final.parent).free > archive.stat().st_size
    )
    digest = sha256_file(archive)
    shutil.copyfile(archive, final)
    assert sha256_file(final) == digest
    archive.unlink()
    staging.rmdir()
    report = dict(
        status="verified_expanded_attention_recovery",
        implementation_commit=commit,
        archive={**binding(final), "bytes": final.stat().st_size},
        logical_members=len(members),
        unique_members=len(members) - len(aliases),
        aliases=len(aliases),
        inherited_members=len(inherited),
        qualified_resume_members=len(extra),
        epoch_archive=epochs["archive"],
        epoch_members=len(epochs["files"]),
        retired_epochs=epochs["files"],
        retired_logical_bytes=epochs["logical_bytes"],
        recovered_caches=caches,
        recovered_cache_bytes=sum(c["bytes"] for c in caches),
        epoch_recovery_seconds=epochs["seconds"],
        qualified_resume_seconds=perf_counter() - started,
        initial_full_wall_seconds=None,
        qualification=binding(attempts / "qualification.json"),
        dependencies=dependencies,
        scope="All24 new child fits and96 baseline book records, source overlays/bounds, hedge correction, originals/leads and attempts recover through the two ZIPs and prior diagnostic dependencies. Initial main ZIP ordinary-member verification completed before PickleBuffer byte counting failed; exact recipe/stdout retain that boundary. Only two cache streams resumed, both hash-exact; all new recovery-resume files physically restored/hash-checked. Frozen original project source remains in project/; qualified archiver and resumer are in recovery_resume/. Initial ordinary members were not rebuilt or reverified. Selected/EMA checkpoints, histories, forecasts, caches, accepted stores, old fits and immutable raw sources remain online. Deleted intermediate epochs were verified before retirement.",
    )
    output = Path(run["root"]) / "attention_expansion_recovery.json"
    write_json_atomic(output, report)
    write_json_atomic(PROJECT / "docs/v2_attention_expansion_recovery.json", report)
    run = json.loads(pointer.read_text())
    run["scaling_expanded_recovery"] = binding(output)
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"retired_epochs", "dependencies"}
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
