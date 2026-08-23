from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.modeling.contract import EQUITY_COUNT
from brazil_rv.modeling.data import (
    EXTERNAL_SIDECAR_SCHEMA,
    feature_store_axis_identity,
    feature_store_identity,
    load_external_sidecar,
)
from brazil_rv.preprocessing.external_sidecar_subset import subset_external_sidecar


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_subset_preserves_requested_order_values_masks_and_identity(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    store.mkdir()
    (store / "manifest.json").write_text("{}", encoding="utf-8")
    (store / "feature_schema.json").write_text(
        json.dumps({"contract_version": "M1_FEATURES_PIT_CAUSAL_TOD"}),
        encoding="utf-8",
    )
    (store / "sample_index.parquet").write_bytes(b"sample-index")
    pl.DataFrame({"date_idx": [0], "trade_date": [date(2024, 1, 2)]}).write_parquet(
        store / "date_index.parquet"
    )
    pl.DataFrame(
        {
            "equity_slot": np.arange(EQUITY_COUNT, dtype=np.int16),
            "security_id": [f"SECURITY_{index:03d}" for index in range(EQUITY_COUNT)],
        }
    ).write_parquet(store / "equity_index.parquet")
    source = tmp_path / "source"
    source.mkdir()
    values = np.arange(EQUITY_COUNT * 3, dtype=np.float32).reshape(1, EQUITY_COUNT, 3)
    masks = np.ones_like(values, dtype=bool)
    np.save(source / "values.npy", values, allow_pickle=False)
    np.save(source / "mask.npy", masks, allow_pickle=False)
    manifest = {
        "schema": EXTERNAL_SIDECAR_SCHEMA,
        "cadence": "daily",
        "feature_names": ["first", "second", "third"],
        "feature_store_identity": feature_store_identity(store),
        "axes": feature_store_axis_identity(store),
        "arrays": {
            "values.npy": {
                "shape": list(values.shape),
                "dtype": "float32",
                "sha256": _hash(source / "values.npy"),
            },
            "mask.npy": {
                "shape": list(masks.shape),
                "dtype": "bool",
                "sha256": _hash(source / "mask.npy"),
            },
        },
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    source_identity = load_external_sidecar(source, store).identity
    output = subset_external_sidecar(
        store=store,
        source_dir=source,
        output_dir=tmp_path / "subset",
        features=("third", "first"),
    )

    loaded = load_external_sidecar(output, store)
    assert loaded.feature_names == ("third", "first")
    np.testing.assert_array_equal(np.load(output / "values.npy"), values[..., [2, 0]])
    np.testing.assert_array_equal(np.load(output / "mask.npy"), masks[..., [2, 0]])
    output_manifest = json.loads((output / "manifest.json").read_text())
    assert output_manifest["provenance"]["source_sidecar"] == source_identity
