from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .contract import EXPECTED_EQUITIES, MIN_ACTIVE_EQUITIES

HUMAN_PRIOR_SCHEMA = "B3_HUMAN_PRIORS_V4"
HUMAN_PRIOR_BUILD_MODE = "complete_market_cap_omitted"


@dataclass(frozen=True)
class HumanPriorArtifact:
    pointer: Path
    directory: Path
    pointer_sha256: str
    manifest: dict[str, object]
    metadata_audit: dict[str, object]
    reference_feature_store: Path
    security_metadata: pl.DataFrame
    peer_policy: pl.DataFrame
    selected_relation: NDArray[np.object_]
    selected_group_id: NDArray[np.int32]
    policy_present: NDArray[np.bool_]
    policy_available: NDArray[np.bool_]
    sector_group_id: NDArray[np.int32]
    subsector_group_id: NDArray[np.int32]
    issuer_ids: tuple[str | None, ...]
    ngrd3_slot: int

    def manifest_entry(self) -> dict[str, object]:
        _require(
            sha256_file(self.pointer) == self.pointer_sha256,
            "Human-priors pointer changed during the feature build",
        )
        scope = self.metadata_audit["scope"]
        reference = self.manifest["canonical_inputs"]["feature_store"]
        return {
            "pointer": str(self.pointer),
            "pointer_sha256": self.pointer_sha256,
            "resolved_path": str(self.directory),
            "schema_version": self.manifest["schema_version"],
            "build_mode": self.manifest["build_mode"],
            "canonical_pointer_published": True,
            "classification_snapshot_date": scope["classification_snapshot_date"],
            "classification_is_point_in_time": False,
            "historical_classification_caveat": scope[
                "historical_classification_caveat"
            ],
            "output_sha256": self.manifest["output_sha256"],
            "referenced_feature_store": {
                "resolved_path": reference["resolved_path"],
                "artifact_sha256": reference["artifact_sha256"],
            },
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nullable_ints(values: Sequence[int | None]) -> NDArray[np.int32]:
    return np.asarray(
        [-1 if value is None else int(value) for value in values], dtype=np.int32
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_human_priors(
    pointer: Path,
    resolved_dir: Path,
    market_dates: Sequence[object],
    security_ids: Sequence[str],
) -> HumanPriorArtifact:
    """Validate and load the canonical human-priors sidecar fail-closed."""
    pointer_bytes = pointer.read_bytes()
    pointer_sha256 = hashlib.sha256(pointer_bytes).hexdigest()
    pointer_target = Path(pointer_bytes.decode("utf-8").strip())
    if not pointer_target.is_dir():
        raise FileNotFoundError(f"Human-priors pointer resolves to {pointer_target}")
    _require(
        pointer_target.resolve() == resolved_dir.resolve(),
        "Human-priors pointer changed after canonical input resolution",
    )
    manifest_path = resolved_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest.get("schema_version") == HUMAN_PRIOR_SCHEMA,
        "Wrong human-priors schema",
    )
    _require(
        manifest.get("build_mode") == HUMAN_PRIOR_BUILD_MODE,
        "Human-priors build mode must omit market cap",
    )
    for field, expected in (
        ("market_cap_mode", "omitted"),
        ("market_cap_requested", False),
        ("market_cap_outputs_emitted", False),
        ("market_cap_data_ready", False),
        ("eligible_for_downstream_market_cap_features", False),
        ("canonical_pointer_published", True),
        ("market_cap_evaluation_status", "NOT_EVALUATED"),
        ("market_cap_source_data_ready", None),
        ("raw_market_cap_history_complete", None),
        ("usable_market_cap_history_complete", None),
        ("classification_is_point_in_time", False),
    ):
        _require(
            manifest.get(field) == expected,
            f"Human-priors manifest has invalid {field}",
        )

    output_hashes = manifest.get("output_sha256")
    _require(
        isinstance(output_hashes, dict) and bool(output_hashes),
        "Missing sidecar output hashes",
    )
    required = {
        "metadata_audit.json",
        "security_metadata.parquet",
        "peer_group_index.parquet",
        "peer_policy_security_days.parquet",
    }
    _require(
        required <= set(output_hashes), "Human-priors artifact omits required outputs"
    )
    for filename, expected_hash in output_hashes.items():
        path = resolved_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Human-priors output is missing: {filename}")
        _require(
            sha256_file(path) == expected_hash,
            f"Human-priors hash mismatch: {filename}",
        )

    metadata_audit = json.loads(
        (resolved_dir / "metadata_audit.json").read_text(encoding="utf-8")
    )
    scope = metadata_audit.get("scope", {})
    _require(
        metadata_audit.get("schema_version") == HUMAN_PRIOR_SCHEMA,
        "Audit schema mismatch",
    )
    _require(
        scope.get("accepted_security_count") == EXPECTED_EQUITIES,
        "Audit equity count mismatch",
    )
    _require(
        scope.get("classification_snapshot_date") is not None,
        "Missing classification snapshot",
    )
    _require(bool(scope.get("historical_classification_caveat")), "Missing PIT caveat")

    canonical_inputs = manifest.get("canonical_inputs", {})
    reference = canonical_inputs.get("feature_store", {})
    reference_store = Path(str(reference.get("resolved_path", "")))
    _require(reference_store.is_dir(), "Referenced feature store is unavailable")
    reference_hashes = reference.get("artifact_sha256")
    required_reference = {
        "manifest.json",
        "date_index.parquet",
        "equity_index.parquet",
        "equity_membership.npy",
        "equity_data_ready.npy",
    }
    _require(
        isinstance(reference_hashes, dict)
        and required_reference <= set(reference_hashes),
        "Human-priors lineage omits required feature-store hashes",
    )
    for filename, expected_hash in reference_hashes.items():
        path = reference_store / filename
        _require(
            path.is_file(), f"Referenced feature-store input is missing: {filename}"
        )
        _require(
            sha256_file(path) == expected_hash,
            f"Referenced input hash mismatch: {filename}",
        )

    reference_dates = pl.read_parquet(reference_store / "date_index.parquet").sort(
        "date_idx"
    )
    _require(
        reference_dates["date_idx"].to_list() == list(range(len(market_dates)))
        and tuple(reference_dates["trade_date"]) == tuple(market_dates),
        "Human-priors referenced date axis does not match the build",
    )
    reference_equities = pl.read_parquet(reference_store / "equity_index.parquet").sort(
        "equity_slot"
    )
    _require(
        reference_equities["equity_slot"].to_list() == list(range(len(security_ids)))
        and tuple(reference_equities["security_id"]) == tuple(security_ids),
        "Human-priors referenced equity axis does not match the build",
    )

    security_metadata = pl.read_parquet(
        resolved_dir / "security_metadata.parquet"
    ).sort("equity_slot")
    _require(
        security_metadata.height == EXPECTED_EQUITIES
        and security_metadata["equity_slot"].to_list() == list(range(EXPECTED_EQUITIES))
        and tuple(security_metadata["security_id"]) == tuple(security_ids),
        "Human-priors security metadata does not align by slot and security_id",
    )
    peer_groups = pl.read_parquet(resolved_dir / "peer_group_index.parquet")
    _require(
        peer_groups["peer_group_id"].n_unique() == peer_groups.height,
        "Peer group IDs must be unique",
    )
    allowed_relations = tuple(
        manifest["peer_policy_output_contract"][
            "allowed_non_null_selected_relation_values"
        ]
    )
    _require(set(allowed_relations) == {"SECTOR", "SUBSECTOR"}, "Wrong peer relations")
    _require(
        set(peer_groups["peer_relation"]) <= set(allowed_relations),
        "Peer-group index contains an undeclared relation",
    )
    relation_by_group = dict(
        peer_groups.select("peer_group_id", "peer_relation").iter_rows()
    )
    sector_group_id = _nullable_ints(
        security_metadata["sector_peer_group_id"].to_list()
    )
    subsector_group_id = _nullable_ints(
        security_metadata["subsector_peer_group_id"].to_list()
    )
    for group_id in sector_group_id[sector_group_id >= 0]:
        _require(
            relation_by_group.get(int(group_id)) == "SECTOR", "Invalid sector group ID"
        )
    for group_id in subsector_group_id[subsector_group_id >= 0]:
        _require(
            relation_by_group.get(int(group_id)) == "SUBSECTOR",
            "Invalid subsector group ID",
        )

    peer_policy = pl.read_parquet(resolved_dir / "peer_policy_security_days.parquet")
    availability = peer_policy["peer_policy_available"]
    _require(
        availability.dtype == pl.Boolean and availability.null_count() == 0,
        "Peer-policy availability must be non-null boolean",
    )
    available = peer_policy.filter(pl.col("peer_policy_available"))
    unavailable = peer_policy.filter(~pl.col("peer_policy_available"))
    _require(
        available.height + unavailable.height == peer_policy.height,
        "Peer-policy availability must be non-null boolean",
    )
    peer_policy = peer_policy.sort("date_idx", "equity_slot")
    _require(
        not peer_policy.select("date_idx", "equity_slot").is_duplicated().any(),
        "Duplicate peer-policy (date_idx, equity_slot) key",
    )
    date_values = peer_policy["date_idx"].to_numpy()
    slot_values = peer_policy["equity_slot"].to_numpy()
    _require(
        bool(((date_values >= 0) & (date_values < len(market_dates))).all()),
        "Peer-policy date index is out of bounds",
    )
    _require(
        bool(((slot_values >= 0) & (slot_values < EXPECTED_EQUITIES)).all()),
        "Peer-policy equity slot is out of bounds",
    )
    contract_count = manifest["peer_policy_output_contract"]["security_day_count"]
    _require(peer_policy.height == contract_count, "Peer-policy row count mismatch")

    invalid_available = available.filter(
        pl.col("selected_peer_relation").is_null()
        | ~pl.col("selected_peer_relation").is_in(allowed_relations)
        | pl.col("selected_peer_group_id").is_null()
        | pl.col("selected_other_active_peer_count").is_null()
        | (pl.col("selected_other_active_peer_count") < 1)
        | pl.col("peer_policy_unavailable_reason").is_not_null()
    )
    _require(invalid_available.is_empty(), "Available peer-policy row is incomplete")
    invalid_unavailable = unavailable.filter(
        pl.col("selected_peer_relation").is_not_null()
        | pl.col("selected_peer_group_id").is_not_null()
        | pl.col("selected_other_active_peer_count").is_not_null()
        | pl.col("peer_policy_unavailable_reason").is_null()
    )
    _require(invalid_unavailable.is_empty(), "Unavailable peer-policy row is malformed")
    joined = available.join(
        peer_groups.select("peer_group_id", "peer_relation"),
        left_on="selected_peer_group_id",
        right_on="peer_group_id",
        how="inner",
    )
    _require(
        joined.height == available.height
        and not joined.filter(
            pl.col("selected_peer_relation") != pl.col("peer_relation")
        ).height,
        "Selected peer relation/group ID is inconsistent",
    )
    for row in available.select(
        "equity_slot", "selected_peer_relation", "selected_peer_group_id"
    ).iter_rows(named=True):
        slot = int(row["equity_slot"])
        expected_group = (
            sector_group_id[slot]
            if row["selected_peer_relation"] == "SECTOR"
            else subsector_group_id[slot]
        )
        _require(
            int(row["selected_peer_group_id"]) == int(expected_group),
            "Selected group does not match focal static classification",
        )

    shape = (len(market_dates), EXPECTED_EQUITIES)
    selected_relation = np.full(shape, None, dtype=object)
    selected_group_id = np.full(shape, -1, dtype=np.int32)
    policy_present = np.zeros(shape, dtype=bool)
    policy_available = np.zeros(shape, dtype=bool)
    selected_relation[date_values, slot_values] = peer_policy[
        "selected_peer_relation"
    ].to_numpy()
    selected_group_id[date_values, slot_values] = _nullable_ints(
        peer_policy["selected_peer_group_id"].to_list()
    )
    policy_present[date_values, slot_values] = True
    policy_available[date_values, slot_values] = peer_policy[
        "peer_policy_available"
    ].to_numpy()
    issuer_ids = tuple(
        None if value is None or not str(value).strip() else str(value)
        for value in security_metadata["issuer_id"]
    )
    ngrd3 = security_metadata.filter(pl.col("ticker") == "NGRD3")
    _require(
        ngrd3.height == 1, "Expected NGRD3 as the unique unresolved classification"
    )

    return HumanPriorArtifact(
        pointer=pointer,
        directory=resolved_dir,
        pointer_sha256=pointer_sha256,
        manifest=manifest,
        metadata_audit=metadata_audit,
        reference_feature_store=reference_store,
        security_metadata=security_metadata,
        peer_policy=peer_policy,
        selected_relation=selected_relation,
        selected_group_id=selected_group_id,
        policy_present=policy_present,
        policy_available=policy_available,
        sector_group_id=sector_group_id,
        subsector_group_id=subsector_group_id,
        issuer_ids=issuer_ids,
        ngrd3_slot=int(ngrd3.item(0, "equity_slot")),
    )


def validate_human_prior_reference_inputs(
    artifact: HumanPriorArtifact,
    membership: NDArray[np.bool_],
    readiness: NDArray[np.bool_],
) -> dict[str, bool]:
    reference_membership = np.load(
        artifact.reference_feature_store / "equity_membership.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    reference_readiness = np.load(
        artifact.reference_feature_store / "equity_data_ready.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    membership_equal = np.array_equal(membership, reference_membership)
    readiness_equal = np.array_equal(readiness, reference_readiness)
    _require(
        membership_equal, "Human-priors membership lineage does not match the build"
    )
    _require(readiness_equal, "Human-priors readiness lineage does not match the build")
    active = membership & readiness
    eligible = active.sum(axis=1) >= MIN_ACTIVE_EQUITIES
    expected_policy_rows = active & eligible[:, None]
    _require(
        np.array_equal(artifact.policy_present, expected_policy_rows),
        "Peer-policy security-day keys do not match eligible active securities",
    )
    return {
        "membership_equal": membership_equal,
        "readiness_equal": readiness_equal,
        "policy_security_days_equal": True,
    }


def _chunked_array_equal(left: NDArray[object], right: NDArray[object]) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return all(
        np.array_equal(left[start : start + 16], right[start : start + 16])
        for start in range(0, left.shape[0], 16)
    )


def validate_unchanged_base_outputs(
    artifact: HumanPriorArtifact,
    date_index: pl.DataFrame,
    equity_index: pl.DataFrame,
    membership: NDArray[np.bool_],
    readiness: NDArray[np.bool_],
    targets: NDArray[np.float32],
    label_mask: NDArray[np.bool_],
    sample_index: pl.DataFrame,
) -> dict[str, bool]:
    reference = artifact.reference_feature_store
    checks = {
        "date_index_equal": date_index.equals(
            pl.read_parquet(reference / "date_index.parquet").sort("date_idx")
        ),
        "equity_index_equal": equity_index.equals(
            pl.read_parquet(reference / "equity_index.parquet").sort("equity_slot")
        ),
        "membership_equal": _chunked_array_equal(
            membership,
            np.load(
                reference / "equity_membership.npy", mmap_mode="r", allow_pickle=False
            ),
        ),
        "readiness_equal": _chunked_array_equal(
            readiness,
            np.load(
                reference / "equity_data_ready.npy", mmap_mode="r", allow_pickle=False
            ),
        ),
        "targets_equal": _chunked_array_equal(
            targets,
            np.load(reference / "targets.npy", mmap_mode="r", allow_pickle=False),
        ),
        "label_mask_equal": _chunked_array_equal(
            label_mask,
            np.load(reference / "label_mask.npy", mmap_mode="r", allow_pickle=False),
        ),
        "sample_index_equal": sample_index.equals(
            pl.read_parquet(reference / "sample_index.parquet")
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    _require(not failed, f"Base feature semantics changed: {failed}")
    return checks
