from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from numpy.lib.format import open_memmap

from brazil_rv.modeling.contract import (
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    workspace_path,
)

from .contract import EXPECTED_EQUITIES
from .human_prior_input import load_human_priors
from .intraday_normalization import (
    AFFECTED_DYNAMIC_CHANNELS,
    AFFECTED_PEER_CHANNELS,
    ARMS,
    DECISION_FEATURE_MINUTES,
    INVARIANT_DYNAMIC_CHANNELS,
    VARIANT_SCHEMA,
    VISIBLE_EQUITY_MINUTES,
    build_seasonal_dynamic_features,
    dynamic_validity_from_observed,
    equity_source_hashes,
    iter_reconstructed_equities,
    load_equity_tod_profile,
    load_source_context,
    parent_artifact_hashes,
    parent_identity,
    repository_commit,
    sha256_file,
    validate_equity_tod_profile,
    write_canonical_json,
)
from .peer_features import build_peer_features
from .transforms import add_equity_cross_sectional_dynamic

DYNAMIC_OVERLAY_FILE = "equity_features_overlay.npy"
PEER_OVERLAY_FILE = "equity_peer_features_overlay.npy"
VARIANT_MANIFEST = "intraday_normalization_variant.json"


def _flush(values: np.memmap) -> None:
    values.flush()
    mapping = getattr(values, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _require_equal(actual: np.ndarray, expected: np.ndarray, message: str) -> None:
    if not np.array_equal(actual, expected):
        raise ValueError(message)


def _load_human_prior_artifact(context):
    entry = context.manifest["canonical_inputs"]["human_priors"]
    pointer = workspace_path(entry["pointer"])
    directory = workspace_path(entry["resolved_path"])
    dates = pl.read_parquet(context.parent / "date_index.parquet").sort("date_idx")
    equities = pl.read_parquet(context.parent / "equity_index.parquet").sort(
        "equity_slot"
    )
    return load_human_priors(
        pointer,
        directory,
        tuple(dates.get_column("trade_date")),
        tuple(equities.get_column("security_id")),
        frozen_manifest_entry=entry,
        current_parent_store=context.parent,
    )


def _validate_profile_binding(context, profile_manifest: dict[str, object]) -> None:
    if profile_manifest["parent_feature_store"] != parent_identity(context):
        raise ValueError("Profile is bound to a different parent feature store")
    if profile_manifest["parent_artifact_sha256"] != parent_artifact_hashes(context):
        raise ValueError("Parent feature store changed after profile construction")
    if profile_manifest["equity_source_sha256"] != equity_source_hashes(context):
        raise ValueError("Accepted equity sources changed after profile construction")


def _open_partial_arrays(partial: Path, date_count: int) -> tuple[np.memmap, np.memmap]:
    dynamic = open_memmap(
        partial / DYNAMIC_OVERLAY_FILE,
        mode="w+",
        dtype=np.float32,
        shape=(
            date_count,
            EXPECTED_EQUITIES,
            VISIBLE_EQUITY_MINUTES,
            len(AFFECTED_DYNAMIC_CHANNELS),
        ),
    )
    peer = open_memmap(
        partial / PEER_OVERLAY_FILE,
        mode="w+",
        dtype=np.float32,
        shape=(
            date_count,
            EXPECTED_EQUITIES,
            len(DECISION_FEATURE_MINUTES),
            len(AFFECTED_PEER_CHANNELS),
        ),
    )
    dynamic[...] = 0.0
    peer[...] = 0.0
    return dynamic, peer


def _populate_raw_channels(
    context,
    relative_variance: np.ndarray,
    dynamic_overlays: dict[str, np.memmap],
) -> None:
    parent_ready = np.load(
        context.parent / "equity_data_ready.npy", mmap_mode="r", allow_pickle=False
    )[: context.allowed_date_count]
    parent_dynamic = np.load(
        context.parent / "equity_features.npy", mmap_mode="r", allow_pickle=False
    )
    raw_channels = tuple(
        channel for channel in AFFECTED_DYNAMIC_CHANNELS if channel < 16
    )
    destinations = tuple(
        AFFECTED_DYNAMIC_CHANNELS.index(value) for value in raw_channels
    )
    seen = np.zeros(EXPECTED_EQUITIES, dtype=bool)
    for equity in iter_reconstructed_equities(context):
        if seen[equity.slot]:
            raise ValueError(f"Equity slot {equity.slot} was reconstructed twice")
        seen[equity.slot] = True
        _require_equal(
            equity.data_ready,
            parent_ready[:, equity.slot],
            f"Parent readiness drift for slot {equity.slot}",
        )
        _require_equal(
            equity.dynamic[..., :16],
            parent_dynamic[: context.allowed_date_count, equity.slot, :, :16],
            f"Legacy reconstruction drift for slot {equity.slot}",
        )
        for arm, gamma in ARMS.items():
            if gamma == 0.0:
                continue
            candidate, candidate_valid = build_seasonal_dynamic_features(
                equity.raw_grid,
                equity.observed,
                equity.data_ready,
                equity.sigma,
                relative_variance,
                gamma,
            )
            _require_equal(
                candidate_valid,
                equity.dynamic_valid,
                f"Dynamic validity changed for {arm}, slot {equity.slot}",
            )
            overlay = dynamic_overlays[arm][:, equity.slot]
            for source_channel, destination in zip(
                raw_channels, destinations, strict=True
            ):
                overlay[..., destination] = candidate[
                    :, :VISIBLE_EQUITY_MINUTES, source_channel
                ]
    if not seen.all():
        raise ValueError("Not every parent equity slot was reconstructed")


def _populate_cross_sectional_and_peer_channels(
    context,
    dynamic_overlays: dict[str, np.memmap],
    peer_overlays: dict[str, np.memmap],
) -> None:
    date_count = context.allowed_date_count
    parent_dynamic = np.load(
        context.parent / "equity_features.npy", mmap_mode="r", allow_pickle=False
    )
    parent_peer = np.load(
        context.parent / "equity_peer_features.npy", mmap_mode="r", allow_pickle=False
    )
    parent_peer_valid = np.load(
        context.parent / "equity_peer_valid.npy", mmap_mode="r", allow_pickle=False
    )
    membership = np.load(
        context.parent / "equity_membership.npy", mmap_mode="r", allow_pickle=False
    )
    ready = np.load(
        context.parent / "equity_data_ready.npy", mmap_mode="r", allow_pickle=False
    )
    human = _load_human_prior_artifact(context)
    observed_channel = 5
    raw_channels = tuple(
        channel for channel in AFFECTED_DYNAMIC_CHANNELS if channel < 16
    )
    aggregate_channels = tuple(
        channel for channel in AFFECTED_DYNAMIC_CHANNELS if channel >= 16
    )
    raw_destinations = tuple(
        AFFECTED_DYNAMIC_CHANNELS.index(value) for value in raw_channels
    )
    aggregate_destinations = tuple(
        AFFECTED_DYNAMIC_CHANNELS.index(value) for value in aggregate_channels
    )
    decision_minutes = np.asarray(DECISION_FEATURE_MINUTES, dtype=np.int64)

    for date_idx in range(date_count):
        active = np.asarray(membership[date_idx] & ready[date_idx], dtype=bool)
        parent_day = np.asarray(
            parent_dynamic[date_idx, :, :VISIBLE_EQUITY_MINUTES], dtype=np.float32
        )
        observed = parent_day[..., observed_channel].astype(bool)
        validity = dynamic_validity_from_observed(observed)

        legacy = parent_day.copy()
        add_equity_cross_sectional_dynamic(legacy, validity, active)
        _require_equal(
            legacy[..., aggregate_channels],
            parent_day[..., aggregate_channels],
            f"Legacy cross-sectional reconstruction drift on date {date_idx}",
        )
        legacy_peer = build_peer_features(
            legacy[:, decision_minutes][:, :, (7, 9)],
            validity[:, decision_minutes, :2],
            active,
            human.selected_relation[date_idx],
            human.selected_group_id[date_idx],
            human.sector_group_id,
            human.subsector_group_id,
            human.issuer_ids,
        )
        _require_equal(
            legacy_peer.features,
            parent_peer[date_idx][:, decision_minutes],
            f"Legacy peer reconstruction drift on date {date_idx}",
        )
        _require_equal(
            legacy_peer.valid,
            parent_peer_valid[date_idx][:, decision_minutes],
            f"Legacy peer validity drift on date {date_idx}",
        )

        for arm, gamma in ARMS.items():
            if gamma == 0.0:
                continue
            candidate = parent_day.copy()
            overlay = dynamic_overlays[arm][date_idx]
            for source_channel, destination in zip(
                raw_channels, raw_destinations, strict=True
            ):
                candidate[..., source_channel] = overlay[..., destination]
            add_equity_cross_sectional_dynamic(candidate, validity, active)
            for source_channel, destination in zip(
                aggregate_channels, aggregate_destinations, strict=True
            ):
                overlay[..., destination] = candidate[..., source_channel]
            peer = build_peer_features(
                candidate[:, decision_minutes][:, :, (7, 9)],
                validity[:, decision_minutes, :2],
                active,
                human.selected_relation[date_idx],
                human.selected_group_id[date_idx],
                human.sector_group_id,
                human.subsector_group_id,
                human.issuer_ids,
            )
            _require_equal(
                peer.valid,
                parent_peer_valid[date_idx][:, decision_minutes],
                f"Peer validity changed for {arm} on date {date_idx}",
            )
            peer_overlays[arm][date_idx] = peer.features[..., AFFECTED_PEER_CHANNELS]


def _write_variant_manifest(
    partial: Path,
    arm: str,
    gamma: float,
    context,
    profile_dir: Path,
    profile_manifest: dict[str, object],
) -> None:
    dynamic_path = partial / DYNAMIC_OVERLAY_FILE
    peer_path = partial / PEER_OVERLAY_FILE
    manifest = {
        "schema": VARIANT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": repository_commit(),
        "arm": arm,
        "gamma": gamma,
        "contract_version": context.manifest["contract_version"],
        "profile_estimator_configuration": profile_manifest["configuration"],
        "split_boundaries": {
            "training": profile_manifest["training_window"],
            "validation": profile_manifest["validation_window"],
        },
        "canonical_parent_feature_store": parent_identity(context),
        "parent_artifact_sha256": profile_manifest["parent_artifact_sha256"],
        "profile": {
            "path": str(profile_dir.resolve()),
            "manifest_sha256": sha256_file(profile_dir / "equity_tod_profile.json"),
            "artifact_sha256": profile_manifest["artifacts"],
        },
        "allowed_date_count": context.allowed_date_count,
        "allowed_date_end": str(context.market_dates[context.allowed_date_count - 1]),
        "test_accessed": False,
        "test_rows_present": False,
        "dynamic_overlay": {
            "file": DYNAMIC_OVERLAY_FILE,
            "shape": [
                context.allowed_date_count,
                EXPECTED_EQUITIES,
                VISIBLE_EQUITY_MINUTES,
                len(AFFECTED_DYNAMIC_CHANNELS),
            ],
            "dtype": "float32",
            "channels": list(AFFECTED_DYNAMIC_CHANNELS),
            "sha256": sha256_file(dynamic_path),
        },
        "peer_overlay": {
            "file": PEER_OVERLAY_FILE,
            "shape": [
                context.allowed_date_count,
                EXPECTED_EQUITIES,
                len(DECISION_FEATURE_MINUTES),
                len(AFFECTED_PEER_CHANNELS),
            ],
            "dtype": "float32",
            "minutes": list(DECISION_FEATURE_MINUTES),
            "channels": list(AFFECTED_PEER_CHANNELS),
            "sha256": sha256_file(peer_path),
        },
        "affected_arrays": {
            "equity_features.npy": list(AFFECTED_DYNAMIC_CHANNELS),
            "equity_peer_features.npy": list(AFFECTED_PEER_CHANNELS),
        },
        "parent_bound_arrays": sorted(
            filename
            for filename in context.manifest["outputs"]
            if filename not in {"equity_features.npy", "equity_peer_features.npy"}
        ),
        "parent_bound_dynamic_channels": list(INVARIANT_DYNAMIC_CHANNELS),
        "parent_bound_peer_channels": [2, 3],
        "profile_freeze_date": profile_manifest["training_profile_freeze_date"],
        "validation_update_rule": profile_manifest["validation_update_rule"],
    }
    write_canonical_json(partial / VARIANT_MANIFEST, manifest)


def build_intraday_normalization_variants(
    parent: Path,
    profile_dir: Path,
    output_base: Path,
) -> dict[str, Path]:
    """Build both candidate overlays in one raw-source pass."""
    context = load_source_context(parent)
    profile_manifest, relative_variance = load_equity_tod_profile(profile_dir)
    _validate_profile_binding(context, profile_manifest)
    output_base.mkdir(parents=True, exist_ok=True)
    arms = tuple(arm for arm, gamma in ARMS.items() if gamma > 0.0)
    final = {arm: output_base / arm for arm in arms}
    partial = {
        arm: path.with_name(f"{path.name}.partial") for arm, path in final.items()
    }
    for path in (*final.values(), *partial.values()):
        if path.exists():
            raise FileExistsError(path)
    dynamic_overlays: dict[str, np.memmap] = {}
    peer_overlays: dict[str, np.memmap] = {}
    published: list[Path] = []
    try:
        for arm in arms:
            partial[arm].mkdir(parents=True)
            dynamic_overlays[arm], peer_overlays[arm] = _open_partial_arrays(
                partial[arm], context.allowed_date_count
            )
        _populate_raw_channels(context, relative_variance, dynamic_overlays)
        _populate_cross_sectional_and_peer_channels(
            context, dynamic_overlays, peer_overlays
        )
        for arm in arms:
            _flush(dynamic_overlays.pop(arm))
            _flush(peer_overlays.pop(arm))
            _write_variant_manifest(
                partial[arm], arm, ARMS[arm], context, profile_dir, profile_manifest
            )
            os.replace(partial[arm], final[arm])
            published.append(final[arm])
    except BaseException:
        for values in (*dynamic_overlays.values(), *peer_overlays.values()):
            try:
                _flush(values)
            except BaseException:
                pass
        for path in partial.values():
            shutil.rmtree(path, ignore_errors=True)
        for path in published:
            shutil.rmtree(path, ignore_errors=True)
        raise
    return final


def validate_intraday_normalization_variant(
    variant: Path,
    expected_arm: str,
    *,
    verify_parent_hashes: bool = True,
) -> dict[str, object]:
    if expected_arm not in tuple(ARMS)[1:]:
        raise ValueError(f"Unsupported candidate arm: {expected_arm}")
    manifest_path = variant / VARIANT_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != VARIANT_SCHEMA:
        raise ValueError("Wrong normalization-variant schema")
    if manifest.get("repository_commit") != repository_commit():
        raise ValueError("Normalization variant repository commit mismatch")
    if (
        manifest.get("arm") != expected_arm
        or manifest.get("gamma") != ARMS[expected_arm]
    ):
        raise ValueError("Normalization variant arm/gamma mismatch")
    if (
        manifest.get("test_accessed") is not False
        or manifest.get("test_rows_present") is not False
    ):
        raise ValueError("Normalization variant is not development-only")
    expected_files = {VARIANT_MANIFEST, DYNAMIC_OVERLAY_FILE, PEER_OVERLAY_FILE}
    if {path.name for path in variant.iterdir()} != expected_files:
        raise ValueError("Normalization variant file inventory is invalid")
    parent = workspace_path(manifest["canonical_parent_feature_store"]["path"])
    context = load_source_context(parent)
    expected_parent = parent_identity(context)
    if manifest.get("canonical_parent_feature_store") != expected_parent:
        raise ValueError("Normalization variant parent identity mismatch")
    if verify_parent_hashes and manifest.get(
        "parent_artifact_sha256"
    ) != parent_artifact_hashes(context):
        raise ValueError("Normalization variant parent hashes mismatch")
    if manifest.get("contract_version") != context.manifest["contract_version"]:
        raise ValueError("Normalization variant contract version mismatch")
    if manifest.get("split_boundaries") != {
        "training": [str(TRAIN_START), str(TRAIN_END)],
        "validation": [str(VALIDATION_START), str(VALIDATION_END)],
    }:
        raise ValueError("Normalization variant split boundaries mismatch")
    if manifest.get("allowed_date_count") != context.allowed_date_count or manifest.get(
        "allowed_date_end"
    ) != str(VALIDATION_END):
        raise ValueError("Normalization variant development date boundary mismatch")
    profile = manifest["profile"]
    profile_dir = Path(profile["path"])
    profile_manifest_path = profile_dir / "equity_tod_profile.json"
    if sha256_file(profile_manifest_path) != profile["manifest_sha256"]:
        raise ValueError("Normalization profile manifest hash mismatch")
    profile_manifest, _ = validate_equity_tod_profile(
        profile_dir, expected_context=context
    )
    if profile_manifest["artifacts"] != profile["artifact_sha256"]:
        raise ValueError("Normalization profile artifact identity mismatch")
    if (
        manifest.get("profile_estimator_configuration")
        != profile_manifest["configuration"]
    ):
        raise ValueError("Normalization estimator configuration mismatch")
    if (
        manifest.get("profile_freeze_date")
        != profile_manifest["training_profile_freeze_date"]
    ):
        raise ValueError("Normalization profile freeze date mismatch")
    if manifest.get("validation_update_rule") != "frozen_training_end_profile":
        raise ValueError("Normalization validation profile rule mismatch")
    expected_dynamic = {
        "file": DYNAMIC_OVERLAY_FILE,
        "shape": [
            context.allowed_date_count,
            EXPECTED_EQUITIES,
            VISIBLE_EQUITY_MINUTES,
            len(AFFECTED_DYNAMIC_CHANNELS),
        ],
        "dtype": "float32",
        "channels": list(AFFECTED_DYNAMIC_CHANNELS),
    }
    expected_peer = {
        "file": PEER_OVERLAY_FILE,
        "shape": [
            context.allowed_date_count,
            EXPECTED_EQUITIES,
            len(DECISION_FEATURE_MINUTES),
            len(AFFECTED_PEER_CHANNELS),
        ],
        "dtype": "float32",
        "minutes": list(DECISION_FEATURE_MINUTES),
        "channels": list(AFFECTED_PEER_CHANNELS),
    }
    for key, expected in (
        ("dynamic_overlay", expected_dynamic),
        ("peer_overlay", expected_peer),
    ):
        entry = manifest.get(key)
        if not isinstance(entry, dict) or any(
            entry.get(name) != value for name, value in expected.items()
        ):
            raise ValueError(f"Normalization overlay metadata mismatch: {key}")
        if set(entry) != {*expected, "sha256"}:
            raise ValueError(f"Normalization overlay metadata has extra fields: {key}")
        path = variant / str(entry["file"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Normalization overlay hash mismatch: {path.name}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(values.shape) != expected["shape"] or values.dtype != np.float32:
            raise ValueError(f"Normalization overlay contract mismatch: {path.name}")
        if not np.isfinite(values).all():
            raise ValueError(f"Normalization overlay is non-finite: {path.name}")
    if manifest.get("affected_arrays") != {
        "equity_features.npy": list(AFFECTED_DYNAMIC_CHANNELS),
        "equity_peer_features.npy": list(AFFECTED_PEER_CHANNELS),
    }:
        raise ValueError("Normalization affected-array contract mismatch")
    expected_parent_bound = sorted(
        filename
        for filename in context.manifest["outputs"]
        if filename not in {"equity_features.npy", "equity_peer_features.npy"}
    )
    if manifest.get("parent_bound_arrays") != expected_parent_bound:
        raise ValueError("Normalization parent-bound array contract mismatch")
    if manifest.get("parent_bound_dynamic_channels") != list(
        INVARIANT_DYNAMIC_CHANNELS
    ):
        raise ValueError("Normalization parent-bound dynamic channels mismatch")
    if manifest.get("parent_bound_peer_channels") != [2, 3]:
        raise ValueError("Normalization parent-bound peer channels mismatch")
    return manifest
