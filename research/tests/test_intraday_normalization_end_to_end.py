from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import brazil_rv.modeling.data as modeling_data
import brazil_rv.preprocessing.intraday_normalization_variants as variants
import brazil_rv.preprocessing.transforms as transforms
from brazil_rv.modeling.contract import (
    BASELINE_TCN_SETTINGS,
    LOCAL_CONTEXT_COUNT,
    SLOW_FEATURE_COUNT,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    architecture_for_model,
)
from brazil_rv.modeling.data import _build_patch_batch
from brazil_rv.modeling.feature_variant import open_variant_arrays
from brazil_rv.modeling.model import build_neural_model, count_trainable_parameters
from brazil_rv.preprocessing.contract import EQUITY_SESSION_MINUTES
from brazil_rv.preprocessing.intraday_normalization import (
    AFFECTED_DYNAMIC_CHANNELS,
    ARMS,
    DECISION_FEATURE_MINUTES,
    INVARIANT_DYNAMIC_CHANNELS,
    PROFILE_BIN_COUNT,
    ReconstructedEquity,
    dynamic_validity_from_observed,
    estimate_causal_profile,
    sha256_file,
)
from brazil_rv.preprocessing.peer_features import build_peer_features
from brazil_rv.preprocessing.transforms import (
    add_equity_cross_sectional_dynamic,
    build_dynamic_features,
)


def _raw_path(increments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    close = np.exp(np.cumsum(increments, axis=1))
    previous = np.concatenate(
        (np.ones((increments.shape[0], 1)), close[:, :-1]), axis=1
    )
    raw = np.zeros((*increments.shape, 5), dtype=np.float64)
    raw[..., 0] = previous
    raw[..., 1] = np.maximum(previous, close) * np.exp(0.0001)
    raw[..., 2] = np.minimum(previous, close) * np.exp(-0.0001)
    raw[..., 3] = close
    raw[..., 4] = 100.0
    return raw, np.ones(increments.shape, dtype=bool)


def test_profile_to_real_overlay_validator_loader_and_model_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    date_count = 2
    parent_date_count = 3
    equity_count = 2
    parent = tmp_path / "parent"
    profile_dir = tmp_path / "profile"
    output = tmp_path / "variants"
    parent.mkdir()
    profile_dir.mkdir()
    dates = (TRAIN_END, VALIDATION_END)
    daily_variance = np.broadcast_to(
        np.linspace(0.5, 1.5, PROFILE_BIN_COUNT),
        (date_count, PROFILE_BIN_COUNT),
    ).copy()
    daily_count = np.full(daily_variance.shape, 50, dtype=np.int64)
    profile = estimate_causal_profile(daily_variance, daily_count, dates)
    np.save(
        profile_dir / "equity_tod_profile.npy",
        profile.relative_variance,
        allow_pickle=False,
    )
    (profile_dir / "equity_tod_profile.csv").write_text("fixture\n", encoding="utf-8")

    ready = np.zeros((parent_date_count, equity_count), dtype=bool)
    ready[1] = True
    membership = np.ones_like(ready)
    parent_dynamic = np.zeros(
        (parent_date_count, equity_count, EQUITY_SESSION_MINUTES, 26),
        dtype=np.float32,
    )
    equities: list[ReconstructedEquity] = []
    for slot in range(equity_count):
        raw, observed = _raw_path(
            np.full(
                (date_count, EQUITY_SESSION_MINUTES),
                0.0001 * (slot + 1),
                dtype=np.float64,
            )
        )
        sigma = np.full(date_count, 0.002, dtype=np.float64)
        dynamic, dynamic_valid = build_dynamic_features(
            raw,
            observed,
            ready[:date_count, slot],
            sigma,
            is_rate=False,
            first_observed_open=True,
        )
        parent_dynamic[:date_count, slot] = dynamic
        equities.append(
            ReconstructedEquity(
                slot,
                f"security-{slot}",
                tmp_path / f"source-{slot}.parquet",
                raw,
                observed,
                dynamic,
                dynamic_valid,
                sigma,
                ready[:date_count, slot],
            )
        )
    monkeypatch.setattr(transforms, "MIN_ACTIVE_EQUITIES", equity_count)
    for date_idx in range(date_count):
        day = parent_dynamic[date_idx, :, : variants.VISIBLE_EQUITY_MINUTES]
        validity = dynamic_validity_from_observed(day[..., 5].astype(bool))
        add_equity_cross_sectional_dynamic(
            day, validity, membership[date_idx] & ready[date_idx]
        )

    human = SimpleNamespace(
        selected_relation=np.full((date_count, equity_count), "SECTOR", dtype=object),
        selected_group_id=np.zeros((date_count, equity_count), dtype=np.int64),
        sector_group_id=np.zeros(equity_count, dtype=np.int64),
        subsector_group_id=np.zeros(equity_count, dtype=np.int64),
        issuer_ids=("issuer", "issuer"),
    )
    parent_peer = np.zeros(
        (parent_date_count, equity_count, EQUITY_SESSION_MINUTES, 6), dtype=np.float32
    )
    parent_peer_valid = np.zeros(
        (parent_date_count, equity_count, EQUITY_SESSION_MINUTES, 4), dtype=bool
    )
    decisions = np.asarray(DECISION_FEATURE_MINUTES, dtype=np.int64)
    for date_idx in range(date_count):
        day = parent_dynamic[date_idx, :, : variants.VISIBLE_EQUITY_MINUTES]
        validity = dynamic_validity_from_observed(day[..., 5].astype(bool))
        peer = build_peer_features(
            day[:, decisions][:, :, (7, 9)],
            validity[:, decisions, :2],
            membership[date_idx] & ready[date_idx],
            human.selected_relation[date_idx],
            human.selected_group_id[date_idx],
            human.sector_group_id,
            human.subsector_group_id,
            human.issuer_ids,
        )
        parent_peer[date_idx][:, decisions] = peer.features
        parent_peer_valid[date_idx][:, decisions] = peer.valid
    np.save(parent / "equity_features.npy", parent_dynamic, allow_pickle=False)
    np.save(parent / "equity_peer_features.npy", parent_peer, allow_pickle=False)
    np.save(parent / "equity_peer_valid.npy", parent_peer_valid, allow_pickle=False)
    np.save(parent / "equity_membership.npy", membership, allow_pickle=False)
    np.save(parent / "equity_data_ready.npy", ready, allow_pickle=False)
    np.save(
        parent / "equity_slow.npy",
        np.ones(
            (parent_date_count, equity_count, SLOW_FEATURE_COUNT), dtype=np.float32
        ),
        allow_pickle=False,
    )
    context_features = np.zeros(
        (parent_date_count, LOCAL_CONTEXT_COUNT, 465, 26), dtype=np.float32
    )
    np.save(parent / "context_features.npy", context_features, allow_pickle=False)

    outputs = {
        name: {}
        for name in (
            "equity_features.npy",
            "equity_peer_features.npy",
            "equity_peer_valid.npy",
            "equity_membership.npy",
            "equity_data_ready.npy",
            "equity_slow.npy",
            "context_features.npy",
        )
    }
    context = SimpleNamespace(
        parent=parent,
        allowed_date_count=date_count,
        market_dates=dates,
        manifest={"contract_version": "fixture", "outputs": outputs},
    )
    parent_identity = {
        "path": str(parent.resolve()),
        "contract_version": "fixture",
        "metadata_sha256": "development-parent",
        "hash_scope": {"kind": "development_only", "end_date": str(VALIDATION_END)},
    }
    parent_hashes = {"scope": "development-only"}
    source_hashes = {"scope": "canonical-development-rows"}
    profile_manifest = {
        "parent_feature_store": parent_identity,
        "parent_artifact_sha256": parent_hashes,
        "equity_source_sha256": source_hashes,
        "configuration": {"bin_minutes": 30},
        "training_window": [str(TRAIN_START), str(TRAIN_END)],
        "validation_window": [str(VALIDATION_START), str(VALIDATION_END)],
        "training_profile_freeze_date": str(TRAIN_END),
        "validation_update_rule": "frozen_training_end_profile",
        "artifacts": {"equity_tod_profile.npy": "fixture"},
    }
    (profile_dir / "equity_tod_profile.json").write_text(
        json.dumps(profile_manifest), encoding="utf-8"
    )
    monkeypatch.setattr(variants, "EXPECTED_EQUITIES", equity_count)
    monkeypatch.setattr(variants, "load_source_context", lambda _parent: context)
    monkeypatch.setattr(
        variants,
        "load_equity_tod_profile",
        lambda _profile: (profile_manifest, profile.relative_variance),
    )
    monkeypatch.setattr(
        variants,
        "validate_equity_tod_profile",
        lambda _profile, expected_context: (
            profile_manifest,
            profile.relative_variance,
        ),
    )
    monkeypatch.setattr(variants, "parent_identity", lambda _context: parent_identity)
    monkeypatch.setattr(
        variants, "parent_artifact_hashes", lambda _context: parent_hashes
    )
    monkeypatch.setattr(
        variants, "equity_source_hashes", lambda _context: source_hashes
    )
    monkeypatch.setattr(
        variants, "iter_reconstructed_equities", lambda _context: iter(equities)
    )
    monkeypatch.setattr(variants, "_load_human_prior_artifact", lambda _context: human)

    built = variants.build_intraday_normalization_variants(parent, profile_dir, output)

    for arm in ("equity_tod_half", "equity_tod_full"):
        manifest = variants.validate_intraday_normalization_variant(built[arm], arm)
        assert {22, 23} <= set(manifest["dynamic_overlay"]["channels"])
        assert {18, 19} <= set(manifest["parent_bound_dynamic_channels"])
        arrays = open_variant_arrays(
            built[arm],
            parent,
            manifest,
            ("equity_features.npy", "context_features.npy"),
        )
        equity = arrays["equity_features.npy"]
        selected = equity[np.asarray([1]), :, :15, :]
        overlay = equity.overlay[1:2, :, :15]
        for channel in AFFECTED_DYNAMIC_CHANNELS:
            position = AFFECTED_DYNAMIC_CHANNELS.index(channel)
            assert np.array_equal(selected[..., channel], overlay[..., position])
        for channel in INVARIANT_DYNAMIC_CHANNELS:
            assert np.array_equal(
                selected[..., channel], parent_dynamic[1:2, :, :15, channel]
            )
        assert np.array_equal(arrays["context_features.npy"], context_features)
        with pytest.raises(ValueError, match="held-out"):
            equity[np.asarray([2]), :, :15, :]

    monkeypatch.setattr(modeling_data, "EQUITY_COUNT", equity_count)
    legacy_arrays = {
        "equity_features.npy": parent_dynamic,
        "equity_slow.npy": np.load(parent / "equity_slow.npy"),
    }
    full_manifest = variants.validate_intraday_normalization_variant(
        built["equity_tod_full"], "equity_tod_full"
    )
    candidate_arrays = {
        **legacy_arrays,
        **open_variant_arrays(
            built["equity_tod_full"],
            parent,
            full_manifest,
            ("equity_features.npy",),
        ),
    }
    arguments = (
        np.asarray([1]),
        np.asarray([15]),
        np.asarray([0]),
        np.asarray([75]),
        np.ones((1, equity_count), dtype=bool),
        None,
    )
    legacy_batch = _build_patch_batch(legacy_arrays, *arguments)
    candidate_batch = _build_patch_batch(candidate_arrays, *arguments)
    assert candidate_batch["patches"].shape == legacy_batch["patches"].shape
    assert candidate_batch["patches"].dtype == np.float32
    assert np.array_equal(
        candidate_batch["instrument_mask"], legacy_batch["instrument_mask"]
    )
    assert np.array_equal(
        candidate_batch["slow_features"], legacy_batch["slow_features"]
    )
    architecture = architecture_for_model("tcn", BASELINE_TCN_SETTINGS)
    counts = {
        count_trainable_parameters(build_neural_model("tcn", architecture, "selected"))
        for _arm in ARMS
    }
    assert len(counts) == 1
    assert (
        sha256_file(profile_dir / "equity_tod_profile.json")
        == full_manifest["profile"]["manifest_sha256"]
    )
    for arm, gamma in ARMS.items():
        assert gamma in (0.0, 0.5, 1.0)
        if arm != "legacy_daily_vol":
            assert built[arm].is_dir()
