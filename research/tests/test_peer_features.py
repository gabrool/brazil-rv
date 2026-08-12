from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from brazil_rv.preprocessing.human_prior_input import (
    load_human_priors,
    validate_human_prior_reference_inputs,
)
from brazil_rv.preprocessing.peer_features import (
    build_peer_features,
    validate_peer_arrays,
)


def _peer_case(equity_count: int = 4, minute_count: int = 3) -> dict[str, object]:
    return {
        "normalized_returns": np.zeros(
            (equity_count, minute_count, 2), dtype=np.float32
        ),
        "return_valid": np.ones((equity_count, minute_count, 2), dtype=bool),
        "active": np.ones(equity_count, dtype=bool),
        "selected_relation": np.full(equity_count, "SUBSECTOR", dtype=object),
        "selected_group_id": np.full(equity_count, 20, dtype=np.int32),
        "sector_group_id": np.full(equity_count, 10, dtype=np.int32),
        "subsector_group_id": np.full(equity_count, 20, dtype=np.int32),
        "issuer_ids": (None,) * equity_count,
    }


def _build(case: dict[str, object]):
    return build_peer_features(**case)


def test_subsector_selection_self_excluded_median_and_centered_rank() -> None:
    case = _peer_case(equity_count=3, minute_count=1)
    case["normalized_returns"][:, 0, 0] = [1.0, 2.0, 4.0]
    result = _build(case)
    assert result.valid[:, 0, 0].all()
    np.testing.assert_allclose(result.features[:, 0, 0], [-2.0, -0.5, 2.5])
    np.testing.assert_allclose(result.features[:, 0, 2], [-2.0 / 3.0, 0.0, 2.0 / 3.0])
    np.testing.assert_array_equal(result.usable_peer_count[:, 0, 0], 2)


def test_sector_fallback_uses_candidates_regardless_of_their_own_policy() -> None:
    case = _peer_case(equity_count=3, minute_count=1)
    case["selected_relation"] = np.array(["SECTOR", "SUBSECTOR", None], dtype=object)
    case["selected_group_id"] = np.array([10, 20, -1], dtype=np.int32)
    case["normalized_returns"][:, 0, 0] = [1.0, 3.0, 5.0]
    result = _build(case)
    assert result.valid[0, 0, 0]
    assert result.features[0, 0, 0] == -3.0
    assert result.usable_peer_count[0, 0, 0] == 2


def test_centered_midranks_average_ties() -> None:
    case = _peer_case(equity_count=3, minute_count=1)
    case["normalized_returns"][:, 0, 0] = [1.0, 1.0, 3.0]
    result = _build(case)
    np.testing.assert_allclose(
        result.features[:, 0, 2], [-1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0]
    )


def test_selected_peers_require_two_other_valid_securities() -> None:
    case = _peer_case(equity_count=3, minute_count=1)
    case["return_valid"][2, 0, 0] = False
    result = _build(case)
    assert not result.valid[:, 0, 0].any()
    assert not result.features[:, 0, [0, 2]].any()


def test_same_issuer_with_exactly_one_peer() -> None:
    case = _peer_case(equity_count=2, minute_count=1)
    case["selected_relation"][:] = None
    case["selected_group_id"][:] = -1
    case["issuer_ids"] = ("ISSUER-A", "ISSUER-A")
    case["normalized_returns"][:, 0, 0] = [1.0, 4.0]
    result = _build(case)
    assert result.valid[:, 0, 2].all()
    np.testing.assert_array_equal(result.features[:, 0, 4], [-3.0, 3.0])
    np.testing.assert_array_equal(result.usable_peer_count[:, 0, 2], 1)


def test_same_issuer_uses_median_of_multiple_peers() -> None:
    case = _peer_case(equity_count=4, minute_count=1)
    case["issuer_ids"] = ("ISSUER-A", "ISSUER-A", "ISSUER-A", "ISSUER-B")
    case["normalized_returns"][:, 0, 0] = [1.0, 3.0, 5.0, 9.0]
    result = _build(case)
    assert result.valid[:3, 0, 2].all()
    assert result.features[0, 0, 4] == -3.0
    assert not result.valid[3, 0, 2]


def test_no_or_null_issuer_peer_is_zero_invalid_and_does_not_remove_sample() -> None:
    case = _peer_case(equity_count=30, minute_count=1)
    case["selected_relation"][:] = None
    case["selected_group_id"][:] = -1
    case["issuer_ids"] = tuple([None] + [f"ISSUER-{slot}" for slot in range(1, 30)])
    result = _build(case)
    assert case["active"].sum() == 30
    assert not result.valid[..., 2:].any()
    assert not result.features[..., 4:].any()
    assert result.features.shape[0] == 30


def test_unavailable_classification_keeps_selected_features_zero_invalid() -> None:
    case = _peer_case(equity_count=3, minute_count=1)
    case["selected_relation"][0] = None
    case["selected_group_id"][0] = -1
    result = _build(case)
    assert not result.valid[0, :, :2].any()
    assert not result.features[0, :, :4].any()


def test_inactive_and_unobserved_peers_are_not_usable() -> None:
    case = _peer_case(equity_count=4, minute_count=1)
    case["active"][2] = False
    case["return_valid"][3, 0, 0] = False
    result = _build(case)
    assert not result.valid[:, 0, 0].any()
    case["return_valid"][0, 0, 0] = False
    result = _build(case)
    assert not result.valid[0, 0, 0]


def test_15m_and_60m_validity_are_independent() -> None:
    case = _peer_case(equity_count=3, minute_count=1)
    case["return_valid"][:, 0, 1] = False
    result = _build(case)
    assert result.valid[:, 0, 0].all()
    assert not result.valid[:, 0, 1].any()
    assert result.features[:, 0, [1, 3]].sum() == 0.0


def test_future_mutation_cannot_change_current_or_earlier_peer_features() -> None:
    case = _peer_case(equity_count=3, minute_count=4)
    case["normalized_returns"][:, :, 0] = np.array(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [4.0, 5.0, 6.0, 7.0]]
    )
    baseline = _build(case)
    changed = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in case.items()
    }
    changed["normalized_returns"][:, 3] = 1_000.0
    mutated = _build(changed)
    np.testing.assert_array_equal(mutated.features[:, :3], baseline.features[:, :3])
    np.testing.assert_array_equal(mutated.valid[:, :3], baseline.valid[:, :3])


def test_nonfinite_source_returns_are_rejected() -> None:
    case = _peer_case(equity_count=3, minute_count=1)
    case["normalized_returns"][1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _build(case)


def test_peer_array_integrity_rejects_nonzero_false_mask() -> None:
    features = np.zeros((1, 3, 2, 6), dtype=np.float32)
    valid = np.zeros((1, 3, 2, 4), dtype=bool)
    validate_peer_arrays(features, valid)
    features[0, 0, 0, 2] = 0.25
    with pytest.raises(ValueError, match="exactly zero"):
        validate_peer_arrays(features, valid)
    features[...] = 0.0
    valid[0, 0, 0, 0] = True
    features[0, 0, 0, 2] = 1.0
    with pytest.raises(ValueError, match="rank outside"):
        validate_peer_arrays(features, valid)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sidecar(tmp_path: Path, *, duplicate_policy: bool = False):
    security_ids = tuple(f"SEC-{slot:03d}" for slot in range(158))
    trade_dates = (date(2024, 1, 2),)
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "manifest.json").write_text("{}", encoding="utf-8")
    pl.DataFrame(
        {"date_idx": pl.Series([0], dtype=pl.Int32), "trade_date": trade_dates}
    ).write_parquet(reference / "date_index.parquet")
    pl.DataFrame(
        {
            "equity_slot": pl.Series(range(158), dtype=pl.Int16),
            "security_id": security_ids,
        }
    ).write_parquet(reference / "equity_index.parquet")
    membership = np.ones((1, 158), dtype=bool)
    readiness = np.ones_like(membership)
    np.save(reference / "equity_membership.npy", membership)
    np.save(reference / "equity_data_ready.npy", readiness)

    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    metadata = pl.DataFrame(
        {
            "equity_slot": pl.Series(range(158), dtype=pl.Int16),
            "security_id": security_ids,
            "ticker": [f"T{slot:03d}" for slot in range(157)] + ["NGRD3"],
            "issuer_id": [f"ISSUER-{slot:03d}" for slot in range(158)],
            "sector_peer_group_id": pl.Series([1] * 157 + [None], dtype=pl.Int32),
            "subsector_peer_group_id": pl.Series([2] * 157 + [None], dtype=pl.Int32),
        }
    )
    metadata.write_parquet(sidecar / "security_metadata.parquet")
    pl.DataFrame(
        {
            "peer_group_id": pl.Series([1, 2], dtype=pl.Int32),
            "peer_relation": ["SECTOR", "SUBSECTOR"],
            "peer_group_key": ["sector", "subsector"],
            "sector": ["Sector", "Sector"],
            "subsector": [None, "Subsector"],
        }
    ).write_parquet(sidecar / "peer_group_index.parquet")
    policy_schema = {
        "date_idx": pl.Int32,
        "equity_slot": pl.Int16,
        "selected_peer_relation": pl.String,
        "selected_peer_group_id": pl.Int32,
        "selected_other_active_peer_count": pl.Int32,
        "sector_fallback_used": pl.Boolean,
        "peer_policy_available": pl.Boolean,
        "peer_policy_unavailable_reason": pl.String,
        "same_sector_peer_count": pl.Int32,
        "same_subsector_peer_count": pl.Int32,
    }
    if duplicate_policy:
        policy = pl.DataFrame(
            {
                "date_idx": [0, 0],
                "equity_slot": [157, 157],
                "selected_peer_relation": [None, None],
                "selected_peer_group_id": [None, None],
                "selected_other_active_peer_count": [None, None],
                "sector_fallback_used": [False, False],
                "peer_policy_available": [False, False],
                "peer_policy_unavailable_reason": [
                    "UNRESOLVED_CLASSIFICATION",
                    "UNRESOLVED_CLASSIFICATION",
                ],
                "same_sector_peer_count": [0, 0],
                "same_subsector_peer_count": [0, 0],
            },
            schema=policy_schema,
        )
    else:
        policy = pl.DataFrame(schema=policy_schema)
    policy.write_parquet(sidecar / "peer_policy_security_days.parquet")
    audit = {
        "schema_version": "B3_HUMAN_PRIORS_V4",
        "scope": {
            "accepted_security_count": 158,
            "classification_snapshot_date": "2026-08-11",
            "historical_classification_caveat": "Frozen current B3 snapshot.",
        },
    }
    (sidecar / "metadata_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    output_names = (
        "metadata_audit.json",
        "security_metadata.parquet",
        "peer_group_index.parquet",
        "peer_policy_security_days.parquet",
    )
    reference_names = (
        "manifest.json",
        "date_index.parquet",
        "equity_index.parquet",
        "equity_membership.npy",
        "equity_data_ready.npy",
    )
    manifest = {
        "schema_version": "B3_HUMAN_PRIORS_V4",
        "build_mode": "complete_market_cap_omitted",
        "market_cap_mode": "omitted",
        "market_cap_requested": False,
        "market_cap_evaluation_status": "NOT_EVALUATED",
        "market_cap_source_data_ready": None,
        "raw_market_cap_history_complete": None,
        "usable_market_cap_history_complete": None,
        "market_cap_outputs_emitted": False,
        "market_cap_data_ready": False,
        "eligible_for_downstream_market_cap_features": False,
        "canonical_pointer_published": True,
        "classification_is_point_in_time": False,
        "canonical_inputs": {
            "feature_store": {
                "resolved_path": str(reference),
                "artifact_sha256": {
                    name: _sha256(reference / name) for name in reference_names
                },
            }
        },
        "peer_policy_output_contract": {
            "allowed_non_null_selected_relation_values": ["SECTOR", "SUBSECTOR"],
            "security_day_count": policy.height,
        },
        "output_sha256": {name: _sha256(sidecar / name) for name in output_names},
    }
    (sidecar / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pointer = tmp_path / "human_priors_pointer.txt"
    pointer.write_text(str(sidecar), encoding="utf-8")
    return pointer, sidecar, trade_dates, security_ids, membership, readiness


@pytest.mark.parametrize("failure", ("hash", "schema", "axis"))
def test_human_prior_hash_schema_and_axis_mismatches_fail_closed(
    tmp_path: Path, failure: str
) -> None:
    pointer, sidecar, trade_dates, security_ids, _, _ = _write_sidecar(tmp_path)
    if failure == "hash":
        with (sidecar / "metadata_audit.json").open("a", encoding="utf-8") as target:
            target.write(" ")
        expected = "hash mismatch"
    elif failure == "schema":
        manifest_path = sidecar / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = "B3_HUMAN_PRIORS_V3"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        expected = "Wrong human-priors schema"
    else:
        security_ids = ("WRONG", *security_ids[1:])
        expected = "referenced equity axis"
    with pytest.raises(ValueError, match=expected):
        load_human_priors(pointer, sidecar, trade_dates, security_ids)


def test_duplicate_peer_policy_key_fails_closed(tmp_path: Path) -> None:
    pointer, sidecar, trade_dates, security_ids, _, _ = _write_sidecar(
        tmp_path, duplicate_policy=True
    )
    with pytest.raises(ValueError, match="Duplicate peer-policy"):
        load_human_priors(pointer, sidecar, trade_dates, security_ids)


def test_null_peer_policy_availability_fails_closed(tmp_path: Path) -> None:
    pointer, sidecar, trade_dates, security_ids, _, _ = _write_sidecar(tmp_path)
    policy_path = sidecar / "peer_policy_security_days.parquet"
    policy_schema = pl.read_parquet_schema(policy_path)
    pl.DataFrame(
        {
            "date_idx": [0],
            "equity_slot": [157],
            "selected_peer_relation": [None],
            "selected_peer_group_id": [None],
            "selected_other_active_peer_count": [None],
            "sector_fallback_used": [False],
            "peer_policy_available": [None],
            "peer_policy_unavailable_reason": ["UNRESOLVED_CLASSIFICATION"],
            "same_sector_peer_count": [0],
            "same_subsector_peer_count": [0],
        },
        schema=policy_schema,
    ).write_parquet(policy_path)
    manifest_path = sidecar / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["peer_policy_output_contract"]["security_day_count"] = 1
    manifest["output_sha256"][policy_path.name] = _sha256(policy_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Peer-policy availability must be non-null boolean",
    ):
        load_human_priors(pointer, sidecar, trade_dates, security_ids)


def test_reference_membership_or_readiness_mismatch_fails_before_use(
    tmp_path: Path,
) -> None:
    pointer, sidecar, trade_dates, security_ids, membership, readiness = _write_sidecar(
        tmp_path
    )
    artifact = load_human_priors(pointer, sidecar, trade_dates, security_ids)
    with pytest.raises(ValueError, match="membership lineage"):
        validate_human_prior_reference_inputs(artifact, ~membership, readiness)
