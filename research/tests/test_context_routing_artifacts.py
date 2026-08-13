from __future__ import annotations

from pathlib import Path

import pytest

from brazil_rv.modeling import context_routing_artifacts as artifacts
from brazil_rv.modeling.context_routing_artifacts import (
    create_validated_archive,
    sha256_file,
    publish_output_pointer,
    validate_archive,
    validate_archive_sha256,
    write_archive_sha256,
)


def test_archive_is_explicit_validated_hashed_and_published_atomically(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "state.json").write_text("state\n", encoding="utf-8")
    nested = output / "audit"
    nested.mkdir()
    (nested / "report.json").write_text("audit\n", encoding="utf-8")
    members = ("state.json", "audit/report.json")
    archive = output / "outputs.zip"
    expected = create_validated_archive(output, members, archive)
    assert validate_archive(archive, expected) == expected
    sidecar = output / "outputs.zip.sha256"
    digest = write_archive_sha256(archive, sidecar)
    assert validate_archive_sha256(archive, sidecar) == digest

    pointer = tmp_path / "_ops" / "current.txt"
    publish_output_pointer(pointer, output, archive, sidecar, expected)
    assert Path(pointer.read_text(encoding="utf-8").strip()).samefile(output)
    assert not pointer.with_name("current.txt.tmp").exists()


def test_completed_archive_restart_recovers_a_missing_sidecar(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    archive = output / "context_routing_sequence_outputs.zip"
    archive.write_bytes(b"archive")
    sidecar = output / "context_routing_sequence_outputs.zip.sha256"
    digest = write_archive_sha256(archive, sidecar)
    assert digest == sha256_file(archive)
    sidecar.unlink()
    assert not sidecar.exists()
    write_archive_sha256(archive, sidecar)
    assert validate_archive_sha256(archive, sidecar) == digest


@pytest.mark.parametrize(
    "members",
    [
        ("missing.json",),
        ("../escape.json",),
        ("state.json", "state.json"),
        ("state.json", "STATE.JSON"),
        ("/absolute.json",),
        (r"audit\report.json",),
    ],
)
def test_archive_rejects_missing_escape_and_duplicate_members(
    tmp_path: Path, members: tuple[str, ...]
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "state.json").write_text("state", encoding="utf-8")
    with pytest.raises((FileNotFoundError, ValueError)):
        create_validated_archive(output, members, output / "outputs.zip")


def test_archive_rejects_symlink_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    link = output / "link.json"
    link.write_text("not really a link", encoding="utf-8")
    original = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda self: self == link or original(self),
    )
    with pytest.raises(ValueError, match="symlink"):
        create_validated_archive(output, ("link.json",), output / "outputs.zip")


def test_archive_rejects_inputs_that_mutate_during_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    source = output / "state.json"
    source.write_text("state", encoding="utf-8")
    actual_sha = artifacts.sha256_file
    calls = 0

    def unstable(path: Path) -> str:
        nonlocal calls
        calls += 1
        digest = actual_sha(path)
        return digest if calls == 1 else "0" * 64

    monkeypatch.setattr(artifacts, "sha256_file", unstable)
    with pytest.raises(RuntimeError, match="mutated"):
        create_validated_archive(output, ("state.json",), output / "outputs.zip")
    assert not (output / "outputs.zip.tmp").exists()


def test_pointer_is_not_published_when_archive_validation_fails(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    source = output / "state.json"
    source.write_text("state", encoding="utf-8")
    archive = output / "outputs.zip"
    expected = create_validated_archive(output, ("state.json",), archive)
    sidecar = output / "outputs.zip.sha256"
    write_archive_sha256(archive, sidecar)
    source.write_text("changed", encoding="utf-8")
    pointer = tmp_path / "current.txt"
    bad_expected = {"state.json": artifacts.sha256_file(source)}
    assert bad_expected != expected
    with pytest.raises(ValueError, match="hashes"):
        publish_output_pointer(pointer, output, archive, sidecar, bad_expected)
    assert not pointer.exists()
