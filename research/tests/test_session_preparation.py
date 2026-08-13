from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from brazil_rv.modeling.data import CacheWarmupReport
from brazil_rv.modeling.run_profiles import RunProfile
from brazil_rv.modeling import session_preparation as preparation


def _profile(identity: str = "profile-hash") -> RunProfile:
    return RunProfile(
        name="experiment",
        equity_slots=(3, 7),
        security_ids=("SEC3", "SEC7"),
        symbols=("EQ3", "EQ7"),
        decision_indices=(0, 3),
        maximum_epochs=3,
        minimum_active_equities=1,
        minimum_training_dates=1,
        decision_grouped_batches=True,
        provenance={"source": "synthetic"},
        selection=(),
        identity_sha256=identity,
    )


def test_session_preparation_reuse_is_exact_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "store"
    store.mkdir()
    rows = pl.DataFrame({"sample_id": [0, 1, 2]})
    snapshot = {"manifest.json": {"size": 10, "mtime_ns": 20, "sha256": "a"}}
    environment = {
        "boot_identity": "boot-a",
        "hostname": "gh200",
        "machine": "aarch64",
        "python": "3.12",
        "pytorch": "test",
        "cuda": "test",
        "cuda_available": True,
        "gpu_name": "NVIDIA GH200",
    }
    monkeypatch.setattr(preparation, "validate_feature_store", lambda _: rows)
    monkeypatch.setattr(preparation, "load_sample_index", lambda _: rows)
    monkeypatch.setattr(preparation, "_file_snapshot", lambda _: snapshot)
    monkeypatch.setattr(preparation, "_environment", lambda: environment)
    monkeypatch.setattr(
        preparation,
        "warm_feature_store_cache",
        lambda *_: CacheWarmupReport(bytes_read=123, files_read=4, seconds=0.5),
    )
    artifact = tmp_path / preparation.SESSION_PREPARATION_FILENAME
    profile = _profile()

    preparation.prepare_feature_store_session(artifact, store, "a" * 40, profile)
    loaded, warmup = preparation.validate_session_preparation(
        artifact, store, "a" * 40, profile
    )

    assert loaded.equals(rows)
    assert warmup == CacheWarmupReport(bytes_read=123, files_read=4, seconds=0.5)
    with pytest.raises(ValueError, match="stale or identity-mismatched"):
        preparation.validate_session_preparation(artifact, store, "b" * 40, profile)
    with pytest.raises(ValueError, match="stale or identity-mismatched"):
        preparation.validate_session_preparation(
            artifact, store, "a" * 40, _profile("different-profile")
        )
    monkeypatch.setattr(
        preparation,
        "_file_snapshot",
        lambda _: {"manifest.json": {"size": 11, "mtime_ns": 20, "sha256": "a"}},
    )
    with pytest.raises(ValueError, match="stale or identity-mismatched"):
        preparation.validate_session_preparation(artifact, store, "a" * 40, profile)
