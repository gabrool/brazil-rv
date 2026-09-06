from pathlib import Path
from types import SimpleNamespace

import pytest

import brazil_rv.v2.build_store as build_store_module
import brazil_rv.v2.store as store_module


GIB = 1024**3


def test_available_memory_uses_conservative_commit_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        store_module,
        "_raw_memory_status_bytes",
        lambda: {
            "total_physical_memory_bytes": 32 * GIB,
            "available_physical_memory_bytes": 18 * GIB,
            "commit_limit_bytes": 40 * GIB,
            "available_commit_memory_bytes": 7 * GIB,
        },
    )

    status = store_module.available_memory_status_bytes()

    assert status["available_build_memory_bytes"] == 7 * GIB
    assert status["admission_basis"] == (
        "minimum_of_available_physical_and_available_commit_memory"
    )
    assert status["commit_limit_bytes"] == 40 * GIB


def test_available_memory_uses_physical_headroom_without_commit_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        store_module,
        "_raw_memory_status_bytes",
        lambda: {
            "total_physical_memory_bytes": 32 * GIB,
            "available_physical_memory_bytes": 13 * GIB,
            "commit_limit_bytes": None,
            "available_commit_memory_bytes": None,
        },
    )

    status = store_module.available_memory_status_bytes()

    assert status["available_build_memory_bytes"] == 13 * GIB
    assert status["admission_basis"] == "available_physical_memory"


def test_build_preflight_records_both_budgets_and_rejects_low_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_status = {
        "total_physical_memory_bytes": 32 * GIB,
        "available_physical_memory_bytes": 18 * GIB,
        "commit_limit_bytes": 36 * GIB,
        "available_commit_memory_bytes": 9 * GIB,
        "available_build_memory_bytes": 9 * GIB,
        "admission_basis": (
            "minimum_of_available_physical_and_available_commit_memory"
        ),
    }
    monkeypatch.setattr(
        build_store_module,
        "available_memory_status_bytes",
        lambda: memory_status,
    )
    monkeypatch.setattr(
        build_store_module, "_is_windows_system_drive", lambda path: False
    )
    monkeypatch.setattr(
        build_store_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=100 * GIB),
    )
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "manifest.json").write_bytes(b"x")

    with pytest.raises(
        MemoryError,
        match=r"physical 18\.00 GiB, available commit 9\.00 GiB, commit limit 36\.00",
    ):
        report = build_store_module._build_resource_preflight(
            output_dir=tmp_path / "next",
            previous_store=previous,
        )
        build_store_module._require_build_resource_preflight(report)

    memory_status["available_commit_memory_bytes"] = 12 * GIB
    memory_status["available_build_memory_bytes"] = 12 * GIB
    preflight = build_store_module._build_resource_preflight(
        output_dir=tmp_path / "next",
        previous_store=previous,
    )
    build_store_module._require_build_resource_preflight(preflight)
    assert preflight["minimum_required_bytes"] == 10 * GIB
    assert preflight["minimum_output_drive_free_multiple"] == 3
    assert preflight["previous_store_size_bytes"] == 1
    assert preflight["output_drive_free_bytes"] == 100 * GIB
    assert preflight["violations"] == []
    assert preflight["passed"] is True


def test_build_preflight_rejects_system_drive_staging_and_low_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        build_store_module,
        "available_memory_status_bytes",
        lambda: {
            "total_physical_memory_bytes": 32 * GIB,
            "available_physical_memory_bytes": 16 * GIB,
            "commit_limit_bytes": 40 * GIB,
            "available_commit_memory_bytes": 20 * GIB,
            "available_build_memory_bytes": 16 * GIB,
            "admission_basis": "minimum_of_available_physical_and_available_commit_memory",
        },
    )
    monkeypatch.setattr(
        build_store_module, "_directory_size_bytes", lambda path: 4 * GIB
    )
    monkeypatch.setattr(
        build_store_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=11 * GIB),
    )
    monkeypatch.setattr(
        build_store_module.tempfile, "gettempdir", lambda: r"C:\brazil-rv-temp"
    )
    monkeypatch.delenv("BRAZIL_RV_TEST_SCRATCH", raising=False)
    previous = tmp_path / "previous"
    previous.mkdir()

    preflight = build_store_module._build_resource_preflight(
        output_dir=tmp_path / "next",
        previous_store=previous,
    )

    assert preflight["minimum_output_drive_free_bytes"] == 12 * GIB
    assert "process_temp_workspace" in preflight["windows_system_drive_staging_roots"]
    assert preflight["violations"] == [
        "output_drive_free_below_three_times_previous_store",
        "workspace_or_staging_root_on_windows_system_drive",
    ]
    with pytest.raises(OSError, match="output_drive_free_below_three_times"):
        build_store_module._require_build_resource_preflight(preflight)
