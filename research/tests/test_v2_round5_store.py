from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from brazil_rv.v2 import round5_store as extension
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from test_v2_store import _base_store


def test_decision_alignment_preserves_publication_boundary_without_another_lag():
    dates = [date(2024, 1, day) for day in (2, 3, 4)]
    frame = pl.DataFrame(
        {
            "date": [dates[1]],
            "isin": ["BRTEST"],
            "loan_rate": [0.05],
            "loan_rate_age_sessions": [0.0],
        }
    )
    values, mask, ages = extension.align_family(
        frame, dates, ("BRTEST",), ("loan_rate",)
    )
    changed = frame.with_columns(pl.lit(0.25).alias("loan_rate"))
    second, second_mask, _ = extension.align_family(
        changed, dates, ("BRTEST",), ("loan_rate",)
    )
    np.testing.assert_array_equal(values[:1], second[:1])
    np.testing.assert_array_equal(mask, second_mask)
    assert values[1, 0, 0] != second[1, 0, 0]
    assert ages[:, 0, 0].tolist() == [-1, 0, -1]
    assert not mask[2, 0, 0]  # A sparse archive row is not silently carried.
    with pytest.raises(ValueError, match="source-age"):
        extension.align_family(
            frame.drop("loan_rate_age_sessions"), dates, ("BRTEST",), ("loan_rate",)
        )


def test_extension_keeps_protected_files_exact_and_requires_admission_proof(
    tmp_path, monkeypatch
):
    base = _base_store(tmp_path / "base")
    registration = tmp_path / "registration.json"
    write_json_atomic(
        registration,
        {
            "base_store": {
                "root": str(base),
                "manifest_sha256": sha256_file(base / "manifest.json"),
            }
        },
    )
    raw = tmp_path / "events.parquet"
    dates = np.load(base / "date_index.npy", allow_pickle=False).astype(object)
    isins = np.load(base / "isin_index.npy", allow_pickle=False).tolist()
    pl.DataFrame(
        {
            "date": [dates[20]],
            "isin": [isins[0]],
            "sessions_since_financial_filing": [0.0],
            "sessions_since_financial_filing_age_sessions": [0.0],
        }
    ).write_parquet(raw)
    manifest, proof = tmp_path / "source.json", tmp_path / "proof.json"
    write_json_atomic(manifest, {"source": "minute-dated fixture"})
    write_json_atomic(
        proof,
        {
            "data_sha256": sha256_file(raw),
            "causality_passed": True,
            "first_available_decision_passed": True,
        },
    )

    def bind(path: Path):
        return {"path": str(path), "sha256": sha256_file(path)}

    plan = tmp_path / "plan.json"
    payload = {
        "registration": bind(registration),
        "families": [
            {
                "family": "events",
                "status": "admitted",
                "feature_names": ["sessions_since_financial_filing"],
                "source_manifest": bind(manifest),
                "data": bind(raw),
                "availability_proof": bind(proof),
            }
        ],
    }
    write_json_atomic(plan, payload)
    monkeypatch.setattr(
        extension,
        "_git_identity",
        lambda: {"commit": "1" * 40, "tracked_worktree_clean": True},
    )
    result = extension.build(plan, tmp_path / "extended")
    assert result["protected_arrays_exact"]
    assert result["indices_and_tables_exact"]
    values = np.load(tmp_path / "extended/sidecar_events_valid.npy", allow_pickle=False)
    assert values[20, 0, 0]
    assert not values[:20].any()
    payload["families"][0]["availability_proof"]["sha256"] = "0" * 64
    write_json_atomic(plan, payload)
    with pytest.raises(ValueError, match="manifest identity"):
        extension.build(plan, tmp_path / "bad_proof")
    assert not (tmp_path / "bad_proof").exists()


def test_extension_rejects_a_heldout_consumer_row_before_alignment():
    frame = pl.DataFrame(
        {"date": [date(2025, 1, 2)], "isin": ["BRTEST"], "loan_rate": [0.05]}
    )
    with pytest.raises(PermissionError, match="held-out"):
        extension.align_family(frame, [date(2024, 12, 30)], ("BRTEST",), ("loan_rate",))
