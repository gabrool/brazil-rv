from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from brazil_rv.modeling.contract import EQUITY_COUNT, EXPECTED_DECISIONS_PER_DATE
from brazil_rv.modeling.data import load_external_sidecar
from brazil_rv.preprocessing.external_sidecar import materialize_external_sidecar


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _feature_store(tmp_path: Path, date_count: int = 3) -> Path:
    store = tmp_path / "store"
    store.mkdir()
    (store / "manifest.json").write_text("{}", encoding="utf-8")
    (store / "feature_schema.json").write_text(
        json.dumps({"contract_version": "M1_FEATURES_PIT_CAUSAL_TOD"}),
        encoding="utf-8",
    )
    (store / "sample_index.parquet").write_bytes(b"sample-index")
    pl.DataFrame(
        {
            "date_idx": np.arange(date_count, dtype=np.int32),
            "trade_date": [
                date(2024, 1, 2) + timedelta(days=index) for index in range(date_count)
            ],
        }
    ).write_parquet(store / "date_index.parquet")
    pl.DataFrame(
        {
            "equity_slot": np.arange(EQUITY_COUNT, dtype=np.int16),
            "security_id": [f"SECURITY_{index:03d}" for index in range(EQUITY_COUNT)],
        }
    ).write_parquet(store / "equity_index.parquet")
    return store


def _write_source(path: Path, frame: pl.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return path


def test_daily_materialization_uses_exact_availability_date_and_identity(
    tmp_path: Path,
) -> None:
    store = _feature_store(tmp_path)
    source_dir = tmp_path / "normalized"
    source_dir.mkdir()
    source_manifest = source_dir / "manifest.json"
    source_manifest.write_text('{"contract":"normalized-test"}', encoding="utf-8")
    source = _write_source(
        source_dir / "features.parquet",
        pl.DataFrame(
            {
                "source_trade_date": [
                    date(2024, 1, 2),
                    date(2024, 1, 3),
                    date(2024, 1, 2),
                    date(2024, 1, 9),
                ],
                "available_date": [
                    date(2024, 1, 3),
                    date(2024, 1, 4),
                    date(2024, 1, 3),
                    date(2024, 1, 10),
                ],
                "security_id": [
                    "SECURITY_000",
                    "SECURITY_001",
                    "OUTSIDE_UNIVERSE",
                    "SECURITY_000",
                ],
                "activity": [2.5, 0.0, 9.0, 8.0],
                "activity_mask": [True, False, True, True],
            },
            schema_overrides={
                "source_trade_date": pl.Date,
                "available_date": pl.Date,
                "activity": pl.Float32,
                "activity_mask": pl.Boolean,
            },
        ),
    )
    output = materialize_external_sidecar(
        store=store,
        source=source,
        output_dir=tmp_path / "sidecar",
        cadence="daily",
        features=["activity"],
        source_date_column="source_trade_date",
    )

    values = np.load(output / "values.npy")
    mask = np.load(output / "mask.npy")
    assert values.shape == (3, EQUITY_COUNT, 1)
    assert not mask[0].any()
    assert values[1, 0, 0] == pytest.approx(2.5)
    assert mask[1, 0, 0]
    assert values[2, 1, 0] == 0
    assert not mask[2, 1, 0]
    assert np.all(values[~mask] == 0)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance"]["normalized_source_sha256"] == _sha256(source)
    assert manifest["provenance"]["normalized_source_manifest_sha256"] == _sha256(
        source_manifest
    )
    assert manifest["coverage"] == {
        "canonical_cell_count": 3 * EQUITY_COUNT,
        "first_joined_date": "2024-01-03",
        "joined_key_fraction": 2 / (3 * EQUITY_COUNT),
        "joined_row_count": 2,
        "last_joined_date": "2024-01-04",
        "matched_security_count": 2,
        "source_row_count": 4,
        "unmatched_date_row_count": 1,
        "unmatched_either_row_count": 2,
        "unmatched_security_row_count": 1,
        "valid_count_by_feature": {"activity": 1},
        "valid_fraction_by_feature": {"activity": 1 / (3 * EQUITY_COUNT)},
    }
    assert load_external_sidecar(output, store).feature_names == ("activity",)


def test_materializer_rejects_future_observation_duplicate_and_nonzero_invalid(
    tmp_path: Path,
) -> None:
    store = _feature_store(tmp_path)
    base = pl.DataFrame(
        {
            "source_trade_date": [date(2024, 1, 2)],
            "available_date": [date(2024, 1, 3)],
            "security_id": ["SECURITY_000"],
            "feature": [1.0],
            "feature_mask": [True],
        },
        schema_overrides={
            "source_trade_date": pl.Date,
            "available_date": pl.Date,
            "feature": pl.Float32,
            "feature_mask": pl.Boolean,
        },
    )

    future = base.with_columns(
        pl.lit(date(2024, 1, 4), dtype=pl.Date).alias("source_trade_date")
    )
    with pytest.raises(ValueError, match="cannot be later"):
        materialize_external_sidecar(
            store=store,
            source=_write_source(tmp_path / "future.parquet", future),
            output_dir=tmp_path / "future-sidecar",
            cadence="daily",
            features=["feature"],
            source_date_column="source_trade_date",
        )

    with pytest.raises(ValueError, match="duplicate keys"):
        materialize_external_sidecar(
            store=store,
            source=_write_source(
                tmp_path / "duplicate.parquet", pl.concat([base, base])
            ),
            output_dir=tmp_path / "duplicate-sidecar",
            cadence="daily",
            features=["feature"],
        )

    invalid = base.with_columns(
        pl.lit(False).alias("feature_mask"), pl.lit(0.5).alias("feature")
    )
    with pytest.raises(ValueError, match="exactly zero"):
        materialize_external_sidecar(
            store=store,
            source=_write_source(tmp_path / "invalid.parquet", invalid),
            output_dir=tmp_path / "invalid-sidecar",
            cadence="daily",
            features=["feature"],
        )


def test_intraday_materialization_enforces_decision_axis(tmp_path: Path) -> None:
    store = _feature_store(tmp_path)
    frame = pl.DataFrame(
        {
            "available_date": [date(2024, 1, 2)],
            "decision_idx": [7],
            "security_id": ["SECURITY_003"],
            "signal": [-0.25],
            "signal_mask": [True],
        },
        schema_overrides={
            "available_date": pl.Date,
            "decision_idx": pl.Int16,
            "signal": pl.Float32,
            "signal_mask": pl.Boolean,
        },
    )
    output = materialize_external_sidecar(
        store=store,
        source=_write_source(tmp_path / "intraday.parquet", frame),
        output_dir=tmp_path / "intraday-sidecar",
        cadence="intraday",
        features=["signal"],
    )
    values = np.load(output / "values.npy")
    mask = np.load(output / "mask.npy")
    assert values.shape == (3, EQUITY_COUNT, EXPECTED_DECISIONS_PER_DATE, 1)
    assert values[0, 3, 7, 0] == pytest.approx(-0.25)
    assert mask.sum() == 1

    outside = frame.with_columns(pl.lit(55, dtype=pl.Int16).alias("decision_idx"))
    with pytest.raises(ValueError, match="canonical 0..54 axis"):
        materialize_external_sidecar(
            store=store,
            source=_write_source(tmp_path / "outside.parquet", outside),
            output_dir=tmp_path / "outside-sidecar",
            cadence="intraday",
            features=["signal"],
        )


def test_materializer_rejects_corrupt_canonical_axis(tmp_path: Path) -> None:
    store = _feature_store(tmp_path)
    dates = pl.read_parquet(store / "date_index.parquet").with_columns(
        pl.when(pl.col("date_idx") == 2)
        .then(3)
        .otherwise(pl.col("date_idx"))
        .alias("date_idx")
    )
    dates.write_parquet(store / "date_index.parquet")
    source = _write_source(
        tmp_path / "source.parquet",
        pl.DataFrame(
            {
                "available_date": [date(2024, 1, 2)],
                "security_id": ["SECURITY_000"],
                "feature": [1.0],
                "feature_mask": [True],
            },
            schema_overrides={
                "available_date": pl.Date,
                "feature": pl.Float32,
                "feature_mask": pl.Boolean,
            },
        ),
    )
    with pytest.raises(ValueError, match="date axis must be contiguous"):
        materialize_external_sidecar(
            store=store,
            source=source,
            output_dir=tmp_path / "sidecar",
            cadence="daily",
            features=["feature"],
        )
