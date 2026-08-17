from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import brazil_rv.preprocessing.intraday_normalization_variants as variants
from brazil_rv.modeling.contract import (
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
)
from brazil_rv.preprocessing.intraday_normalization import (
    AFFECTED_DYNAMIC_CHANNELS,
    AFFECTED_PEER_CHANNELS,
    INVARIANT_DYNAMIC_CHANNELS,
    VARIANT_SCHEMA,
    sha256_file,
)


def _variant(
    path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str | None
) -> None:
    parent = path / "parent"
    profile = path / "profile"
    variant = path / "variant"
    parent.mkdir()
    profile.mkdir()
    variant.mkdir()
    profile_manifest_path = profile / "equity_tod_profile.json"
    profile_manifest_path.write_text("{}", encoding="utf-8")
    date_count = 2
    equity_count = 2
    visible_minutes = 3
    decision_minutes = (0, 2)
    dynamic_path = variant / variants.DYNAMIC_OVERLAY_FILE
    peer_path = variant / variants.PEER_OVERLAY_FILE
    dynamic_dtype = np.float64 if corruption == "dtype" else np.float32
    dynamic_dates = 1 if corruption == "shape" else date_count
    np.save(
        dynamic_path,
        np.zeros(
            (
                dynamic_dates,
                equity_count,
                visible_minutes,
                len(AFFECTED_DYNAMIC_CHANNELS),
            ),
            dtype=dynamic_dtype,
        ),
        allow_pickle=False,
    )
    np.save(
        peer_path,
        np.zeros(
            (
                date_count,
                equity_count,
                len(decision_minutes),
                len(AFFECTED_PEER_CHANNELS),
            ),
            dtype=np.float32,
        ),
        allow_pickle=False,
    )
    outputs = {
        "equity_features.npy": {},
        "equity_peer_features.npy": {},
        "targets.npy": {},
    }
    context = SimpleNamespace(
        parent=parent,
        allowed_date_count=date_count,
        market_dates=(TRAIN_END, VALIDATION_END),
        manifest={"contract_version": "fixture", "outputs": outputs},
    )
    parent_identity = {
        "path": str(parent),
        "contract_version": "fixture",
        "metadata_sha256": "development-parent",
        "hash_scope": {"kind": "development_only"},
    }
    parent_hashes = {"scope": "development_only"}
    profile_manifest = {
        "artifacts": {"profile": "hash"},
        "configuration": {"bin_minutes": 30},
        "training_profile_freeze_date": str(TRAIN_END),
    }
    manifest = {
        "schema": VARIANT_SCHEMA,
        "repository_commit": "fixture-commit",
        "arm": "equity_tod_half",
        "gamma": 0.5,
        "contract_version": "fixture",
        "profile_estimator_configuration": profile_manifest["configuration"],
        "split_boundaries": {
            "training": [str(TRAIN_START), str(TRAIN_END)],
            "validation": [str(VALIDATION_START), str(VALIDATION_END)],
        },
        "canonical_parent_feature_store": parent_identity,
        "parent_artifact_sha256": parent_hashes,
        "profile": {
            "path": str(profile.resolve()),
            "manifest_sha256": sha256_file(profile_manifest_path),
            "artifact_sha256": profile_manifest["artifacts"],
        },
        "allowed_date_count": date_count,
        "allowed_date_end": str(VALIDATION_END),
        "test_accessed": False,
        "test_rows_present": False,
        "dynamic_overlay": {
            "file": variants.DYNAMIC_OVERLAY_FILE,
            "shape": [
                date_count,
                equity_count,
                visible_minutes,
                len(AFFECTED_DYNAMIC_CHANNELS),
            ],
            "dtype": "float32",
            "channels": list(AFFECTED_DYNAMIC_CHANNELS),
            "sha256": sha256_file(dynamic_path),
        },
        "peer_overlay": {
            "file": variants.PEER_OVERLAY_FILE,
            "shape": [
                date_count,
                equity_count,
                len(decision_minutes),
                len(AFFECTED_PEER_CHANNELS),
            ],
            "dtype": "float32",
            "minutes": list(decision_minutes),
            "channels": list(AFFECTED_PEER_CHANNELS),
            "sha256": sha256_file(peer_path),
        },
        "affected_arrays": {
            "equity_features.npy": list(AFFECTED_DYNAMIC_CHANNELS),
            "equity_peer_features.npy": list(AFFECTED_PEER_CHANNELS),
        },
        "parent_bound_arrays": ["targets.npy"],
        "parent_bound_dynamic_channels": list(INVARIANT_DYNAMIC_CHANNELS),
        "parent_bound_peer_channels": [2, 3],
        "profile_freeze_date": str(TRAIN_END),
        "validation_update_rule": "frozen_training_end_profile",
    }
    if corruption == "arm":
        manifest["arm"] = "equity_tod_full"
    elif corruption == "gamma":
        manifest["gamma"] = 1.0
    elif corruption == "channels":
        manifest["dynamic_overlay"]["channels"] = list(AFFECTED_DYNAMIC_CHANNELS[:-1])
    elif corruption == "minutes":
        manifest["peer_overlay"]["minutes"] = [0]
    elif corruption == "boundary":
        manifest["allowed_date_end"] = str(TRAIN_END)
    elif corruption == "hash":
        manifest["dynamic_overlay"]["sha256"] = "0" * 64
    (variant / variants.VARIANT_MANIFEST).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setattr(variants, "EXPECTED_EQUITIES", equity_count)
    monkeypatch.setattr(variants, "VISIBLE_EQUITY_MINUTES", visible_minutes)
    monkeypatch.setattr(variants, "DECISION_FEATURE_MINUTES", decision_minutes)
    monkeypatch.setattr(variants, "repository_commit", lambda: "fixture-commit")
    monkeypatch.setattr(variants, "workspace_path", Path)
    monkeypatch.setattr(variants, "load_source_context", lambda _path: context)
    monkeypatch.setattr(variants, "parent_identity", lambda _context: parent_identity)
    monkeypatch.setattr(
        variants, "parent_artifact_hashes", lambda _context: parent_hashes
    )
    monkeypatch.setattr(
        variants,
        "validate_equity_tod_profile",
        lambda _path, expected_context: (profile_manifest, np.ones((2, 1))),
    )


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("arm", "arm/gamma"),
        ("gamma", "arm/gamma"),
        ("channels", "metadata mismatch"),
        ("minutes", "metadata mismatch"),
        ("boundary", "date boundary"),
        ("shape", "overlay contract mismatch"),
        ("dtype", "overlay contract mismatch"),
        ("hash", "overlay hash mismatch"),
    ),
)
def test_variant_validator_rejects_contract_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
    message: str,
) -> None:
    _variant(tmp_path, monkeypatch, corruption)
    with pytest.raises(ValueError, match=message):
        variants.validate_intraday_normalization_variant(
            tmp_path / "variant", "equity_tod_half"
        )


def test_variant_validator_accepts_exact_expected_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _variant(tmp_path, monkeypatch, None)
    manifest = variants.validate_intraday_normalization_variant(
        tmp_path / "variant", "equity_tod_half"
    )
    assert manifest["arm"] == "equity_tod_half"
