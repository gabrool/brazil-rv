from __future__ import annotations

import json

import pytest

from brazil_rv.modeling.data import (
    FEATURE_STORE_CONTRACT,
    feature_store_identity,
)


def test_feature_store_identity_rejects_stale_contract(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "sample_index.parquet").write_bytes(b"index")
    schema = tmp_path / "feature_schema.json"
    schema.write_text(json.dumps({"contract_version": "OLD_V4"}), encoding="utf-8")
    with pytest.raises(ValueError, match=FEATURE_STORE_CONTRACT):
        feature_store_identity(tmp_path)

    schema.write_text(
        json.dumps({"contract_version": FEATURE_STORE_CONTRACT}),
        encoding="utf-8",
    )
    assert feature_store_identity(tmp_path)["contract_version"] == (
        FEATURE_STORE_CONTRACT
    )
