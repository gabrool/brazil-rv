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

from brazil_rv.modeling.data import load_external_sidecar


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def subset_external_sidecar(
    *,
    store: Path,
    source_dir: Path,
    output_dir: Path,
    features: Sequence[str],
) -> Path:
    """Create an immutable feature-axis subset of an existing PIT sidecar."""
    source = load_external_sidecar(source_dir, store)
    requested = tuple(features)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Subset features must be nonempty and unique")
    missing = sorted(set(requested).difference(source.feature_names))
    if missing:
        raise ValueError(f"Subset features are absent from the source: {missing}")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    indices = np.asarray(
        [source.feature_names.index(feature) for feature in requested], dtype=np.int64
    )
    source_manifest = json.loads(
        (source.path / "manifest.json").read_text(encoding="utf-8")
    )
    source_values = np.load(source.path / "values.npy", mmap_mode="r")
    source_masks = np.load(source.path / "mask.npy", mmap_mode="r")
    shape = (*source_values.shape[:-1], len(requested))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        values_path = temporary / "values.npy"
        masks_path = temporary / "mask.npy"
        values = np.lib.format.open_memmap(
            values_path, mode="w+", dtype=np.float32, shape=shape
        )
        masks = np.lib.format.open_memmap(
            masks_path, mode="w+", dtype=np.bool_, shape=shape
        )
        for date_idx in range(shape[0]):
            values[date_idx] = np.take(source_values[date_idx], indices, axis=-1)
            masks[date_idx] = np.take(source_masks[date_idx], indices, axis=-1)
        values.flush()
        masks.flush()
        del values, masks

        manifest = {
            "schema": source_manifest["schema"],
            "cadence": source.cadence,
            "feature_names": list(requested),
            "feature_store_identity": source_manifest["feature_store_identity"],
            "axes": source_manifest["axes"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "provenance": {
                "operation": "exact_feature_axis_subset",
                "source_sidecar": source.identity,
                "source_feature_indices": indices.tolist(),
            },
            "arrays": {
                "values.npy": {
                    "shape": list(shape),
                    "dtype": "float32",
                    "sha256": _sha256(values_path),
                },
                "mask.npy": {
                    "shape": list(shape),
                    "dtype": "bool",
                    "sha256": _sha256(masks_path),
                },
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except BaseException:
        if temporary.exists() and temporary.parent == output_dir.parent:
            shutil.rmtree(temporary)
        raise
    load_external_sidecar.cache_clear()
    load_external_sidecar(output_dir, store)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Subset an immutable PIT sidecar")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features", nargs="+", required=True)
    print(subset_external_sidecar(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
