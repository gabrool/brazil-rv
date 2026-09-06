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

    with pytest.raises(
        MemoryError,
        match=r"physical 18\.00 GiB, available commit 9\.00 GiB, commit limit 36\.00",
    ):
        build_store_module._build_resource_preflight()

    memory_status["available_commit_memory_bytes"] = 12 * GIB
    memory_status["available_build_memory_bytes"] = 12 * GIB
    preflight = build_store_module._build_resource_preflight()
    assert preflight == {
        **memory_status,
        "minimum_required_bytes": 10 * GIB,
        "passed": True,
    }
