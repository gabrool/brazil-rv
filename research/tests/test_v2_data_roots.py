from __future__ import annotations

import json
from pathlib import Path

import pytest

from brazil_rv.v2.artifacts import sha256_file
from brazil_rv.v2.data_roots import (
    DATA_ROOTS_ENV,
    DATA_ROOTS_SCHEMA,
    resolve_external_file,
)


def test_foreign_absolute_path_resolves_and_remains_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_root = tmp_path / "lambda-data"
    artifact = local_root / "sealed" / "panel.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"sealed-panel")
    record = {
        "path": r"D:\quant-data\sealed\panel.bin",
        "bytes": artifact.stat().st_size,
        "sha256": sha256_file(artifact),
    }
    override = tmp_path / "data_roots.json"
    override.write_text(
        json.dumps(
            {
                "schema": DATA_ROOTS_SCHEMA,
                "roots": {r"D:\quant-data": str(local_root)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(DATA_ROOTS_ENV, str(override))

    resolved, audit = resolve_external_file(record)

    assert resolved == artifact.resolve()
    assert audit.recorded_path == record["path"]
    assert audit.resolved_path == str(artifact.resolve())
    assert audit.override_file == str(override.resolve())
    assert audit.override_file_sha256 == sha256_file(override)
    assert audit.sha256 == record["sha256"]

    artifact.write_bytes(b"changed-pane")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolve_external_file(record)
