from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.modeling.data import (
    EXTERNAL_SIDECAR_SCHEMA,
    feature_store_axis_identity,
    feature_store_identity,
)

MASK_SUFFIX = "_mask"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_source(
    frame: pl.DataFrame,
    *,
    cadence: str,
    features: tuple[str, ...],
    date_column: str,
    source_date_column: str | None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    if cadence not in ("daily", "intraday"):
        raise ValueError("cadence must be daily or intraday")
    if not features or len(set(features)) != len(features):
        raise ValueError("features must be nonempty and unique")

    mask_columns = [f"{feature}{MASK_SUFFIX}" for feature in features]
    keys = [date_column, "security_id"]
    if cadence == "intraday":
        keys.insert(1, "decision_idx")
    required = [*keys, *features, *mask_columns]
    if source_date_column is not None:
        required.append(source_date_column)
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"Normalized source is missing columns: {missing}")
    if frame.is_empty():
        raise ValueError("Normalized source must contain at least one row")
    if frame.schema[date_column] != pl.Date:
        raise ValueError(f"{date_column} must have Polars Date dtype")
    if frame.schema["security_id"] != pl.String:
        raise ValueError("security_id must have Polars String dtype")
    if source_date_column is not None and frame.schema[source_date_column] != pl.Date:
        raise ValueError(f"{source_date_column} must have Polars Date dtype")
    if any(frame.schema[column] != pl.Boolean for column in mask_columns):
        raise ValueError("Every feature mask must have Polars Boolean dtype")
    if any(not frame.schema[feature].is_numeric() for feature in features):
        raise ValueError("Every feature must have a numeric dtype")
    if cadence == "intraday" and not frame.schema["decision_idx"].is_integer():
        raise ValueError("decision_idx must have an integer dtype")

    checked_columns = [*keys, *features, *mask_columns]
    if source_date_column is not None:
        checked_columns.append(source_date_column)
    null_row = pl.any_horizontal(
        [pl.col(column).is_null() for column in checked_columns]
    )
    if frame.select(null_row.any()).item():
        raise ValueError("Normalized source keys, values, and masks cannot be null")
    duplicate = frame.group_by(keys).len().filter(pl.col("len") > 1).head(1)
    if duplicate.height:
        raise ValueError(f"Normalized source contains duplicate keys: {keys}")
    if cadence == "intraday":
        invalid_decision = frame.filter(~pl.col("decision_idx").is_between(0, 54))
        if invalid_decision.height:
            raise ValueError("decision_idx must lie on the canonical 0..54 axis")
    if source_date_column is not None:
        future = frame.filter(pl.col(source_date_column) > pl.col(date_column))
        if future.height:
            raise ValueError(f"{source_date_column} cannot be later than {date_column}")

    for feature, mask_column in zip(features, mask_columns, strict=True):
        if frame.filter(~pl.col(mask_column) & (pl.col(feature) != 0)).height:
            raise ValueError("Invalid normalized source values must be exactly zero")
        if frame.filter(pl.col(mask_column) & ~pl.col(feature).is_finite()).height:
            raise ValueError("Valid normalized source values must be finite")

    values = frame.select(features).cast(pl.Float32).to_numpy()
    masks = frame.select(mask_columns).to_numpy()
    if not np.isfinite(values[masks]).all():
        raise ValueError("Valid normalized source values must be finite")
    if np.any(values[~masks] != 0):
        raise ValueError("Invalid normalized source values must be exactly zero")
    return mask_columns, values, masks


def _write_sidecar(
    temporary: Path,
    *,
    shape: tuple[int, ...],
    frame: pl.DataFrame,
    values: np.ndarray,
    masks: np.ndarray,
    manifest: dict[str, object],
) -> None:
    values_path = temporary / "values.npy"
    mask_path = temporary / "mask.npy"
    output_values = np.lib.format.open_memmap(
        values_path, mode="w+", dtype=np.float32, shape=shape
    )
    output_mask = np.lib.format.open_memmap(
        mask_path, mode="w+", dtype=np.bool_, shape=shape
    )
    output_values[...] = 0
    output_mask[...] = False

    date_idx = frame.get_column("date_idx").to_numpy()
    equity_slot = frame.get_column("equity_slot").to_numpy()
    if len(shape) == 3:
        output_values[date_idx, equity_slot, :] = values
        output_mask[date_idx, equity_slot, :] = masks
    else:
        decision_idx = frame.get_column("decision_idx").to_numpy()
        output_values[date_idx, equity_slot, decision_idx, :] = values
        output_mask[date_idx, equity_slot, decision_idx, :] = masks
    output_values.flush()
    output_mask.flush()
    del output_values, output_mask

    manifest["arrays"] = {
        "values.npy": {
            "shape": list(shape),
            "dtype": "float32",
            "sha256": _sha256(values_path),
        },
        "mask.npy": {
            "shape": list(shape),
            "dtype": "bool",
            "sha256": _sha256(mask_path),
        },
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def materialize_external_sidecar(
    *,
    store: Path,
    source: Path,
    output_dir: Path,
    cadence: str,
    features: Sequence[str],
    date_column: str = "available_date",
    source_date_column: str | None = None,
) -> Path:
    """Materialize one normalized PIT frame on the exact canonical model axes.

    Availability and any source-specific publication lag must already be encoded in
    ``date_column``. Rows are assigned only to that exact date; this function never
    fills backward or forward. Equity membership remains a loader-side mask.
    """
    store = store.resolve()
    source = source.resolve()
    output_dir = output_dir.resolve()
    if not store.is_dir():
        raise FileNotFoundError(store)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    feature_names = tuple(features)
    frame = pl.read_parquet(source)
    mask_columns, source_values, source_masks = _validate_source(
        frame,
        cadence=cadence,
        features=feature_names,
        date_column=date_column,
        source_date_column=source_date_column,
    )

    dates = (
        pl.read_parquet(store / "date_index.parquet")
        .select("date_idx", "trade_date")
        .sort("date_idx")
    )
    equities = (
        pl.read_parquet(store / "equity_index.parquet")
        .select("equity_slot", "security_id")
        .sort("equity_slot")
    )
    axes = feature_store_axis_identity(store)
    known_dates = set(dates.get_column("trade_date").to_list())
    known_securities = set(equities.get_column("security_id").to_list())
    source_dates = frame.get_column(date_column).to_list()
    source_securities = frame.get_column("security_id").to_list()
    date_match = np.fromiter(
        (value in known_dates for value in source_dates), dtype=np.bool_
    )
    security_match = np.fromiter(
        (value in known_securities for value in source_securities), dtype=np.bool_
    )
    aligned = date_match & security_match
    aligned_frame = (
        frame.with_row_index("_source_row")
        .filter(pl.Series("_aligned", aligned))
        .join(
            dates.rename({"trade_date": date_column}),
            on=date_column,
            how="inner",
            validate="m:1",
        )
        .join(equities, on="security_id", how="inner", validate="m:1")
        .sort("_source_row")
    )
    if aligned_frame.is_empty():
        raise ValueError("No normalized source rows align to the canonical axes")
    source_rows = aligned_frame.get_column("_source_row").to_numpy()
    aligned_values = source_values[source_rows]
    aligned_masks = source_masks[source_rows]

    date_count = int(axes["date_count"])
    equity_count = int(axes["equity_count"])
    decision_count = int(axes["decision_count"])
    shape = (date_count, equity_count)
    if cadence == "intraday":
        shape = (*shape, decision_count)
    shape = (*shape, len(feature_names))
    canonical_cells = int(np.prod(shape[:-1], dtype=np.int64))
    joined_dates = aligned_frame.get_column(date_column)
    valid_counts = aligned_masks.sum(axis=0)
    source_manifest = source.parent / "manifest.json"
    provenance: dict[str, object] = {
        "normalized_source_path": str(source),
        "normalized_source_sha256": _sha256(source),
        "date_column": date_column,
        "source_date_column": source_date_column,
        "key_columns": [
            date_column,
            *(["decision_idx"] if cadence == "intraday" else []),
            "security_id",
        ],
        "mask_columns": mask_columns,
        "availability_join": "exact_no_fill",
    }
    if source_manifest.is_file():
        provenance["normalized_source_manifest_path"] = str(source_manifest.resolve())
        provenance["normalized_source_manifest_sha256"] = _sha256(source_manifest)
    manifest: dict[str, object] = {
        "schema": EXTERNAL_SIDECAR_SCHEMA,
        "cadence": cadence,
        "feature_names": list(feature_names),
        "feature_store_identity": feature_store_identity(store),
        "axes": axes,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "coverage": {
            "source_row_count": frame.height,
            "joined_row_count": aligned_frame.height,
            "unmatched_date_row_count": int((~date_match).sum()),
            "unmatched_security_row_count": int((~security_match).sum()),
            "unmatched_either_row_count": int((~aligned).sum()),
            "canonical_cell_count": canonical_cells,
            "joined_key_fraction": aligned_frame.height / canonical_cells,
            "matched_security_count": aligned_frame.get_column(
                "security_id"
            ).n_unique(),
            "first_joined_date": str(joined_dates.min()),
            "last_joined_date": str(joined_dates.max()),
            "valid_count_by_feature": {
                feature: int(valid_counts[index])
                for index, feature in enumerate(feature_names)
            },
            "valid_fraction_by_feature": {
                feature: float(valid_counts[index] / canonical_cells)
                for index, feature in enumerate(feature_names)
            },
        },
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        _write_sidecar(
            temporary,
            shape=shape,
            frame=aligned_frame,
            values=aligned_values,
            masks=aligned_masks,
            manifest=manifest,
        )
        os.replace(temporary, output_dir)
    except BaseException:
        if temporary.exists() and temporary.parent == output_dir.parent:
            shutil.rmtree(temporary)
        raise
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a normalized PIT feature frame on canonical axes."
    )
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cadence", choices=("daily", "intraday"), required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--date-column", default="available_date")
    parser.add_argument("--source-date-column")
    args = parser.parse_args()
    output = materialize_external_sidecar(
        store=args.store,
        source=args.source,
        output_dir=args.output_dir,
        cadence=args.cadence,
        features=args.features,
        date_column=args.date_column,
        source_date_column=args.source_date_column,
    )
    print(output)


if __name__ == "__main__":
    main()
