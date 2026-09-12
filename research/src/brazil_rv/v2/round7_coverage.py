"""Measure losses at the physical-family to ranked-store boundary, without labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from .artifacts import sha256_file, write_json_atomic
from .round5_store import align_family
from .round7_data import registered_sources
from .store import peak_rss_bytes


def audit(output: Path):
    root, manifest, _, accepted = registered_sources()
    source_records = {}
    ancestor = manifest
    while extension := ancestor["metadata"].get("round5_extension"):
        for record in extension["families"]:
            source_records.setdefault(record["family"], record)
        base = extension["base_store"]
        path = Path(base["root"]) / "manifest.json"
        if sha256_file(path) != base["manifest_sha256"]:
            raise ValueError("family ancestry differs from the sealed source")
        ancestor = json.loads(path.read_text(encoding="utf-8"))
    dates = np.load(root / "date_index.npy", allow_pickle=False).astype(object).tolist()
    isins = tuple(np.load(root / "isin_index.npy", allow_pickle=False).tolist())
    active = np.load(root / manifest["arrays"]["active"]["path"], allow_pickle=False)
    specifications = {
        (r["family"], r["name"]): r
        for r in manifest["metadata"]["feature_schema"]["specifications"]
    }
    rows, missing = [], []
    for family_key, fields in manifest["feature_names"].items():
        if not family_key.startswith("sidecar_"):
            continue
        family = family_key.removeprefix("sidecar_")
        record = source_records.get(family)
        if record is None or not record.get("data"):
            missing.append(
                {
                    "family": family,
                    "reason": "No physical family artifact in the sealed extension ancestry; requires its source-specific builder audit.",
                }
            )
            continue
        source = Path(record["data"]["path"])
        if sha256_file(source) != record["data"]["sha256"]:
            raise ValueError(f"physical source differs: {family}")
        frame = pl.read_parquet(source)
        values, valid, ages = align_family(frame, dates, isins, tuple(fields))
        del frame, values, ages
        old = np.load(
            root / manifest["arrays"][family_key + "_valid"]["path"], mmap_mode="r"
        )
        for f, name in enumerate(fields):
            physical = valid[..., f] & active
            lost = physical & ~old[..., f]
            sparse = physical.sum(axis=1) < 20
            spec = specifications[family_key, name]
            rows.append(
                {
                    "family": family,
                    "field": name,
                    "transform": spec["transform"],
                    "physical_active_name_days": int(physical.sum()),
                    "lost_name_days": int(lost.sum()),
                    "lost_on_sparse_dates": int((lost & sparse[:, None]).sum()),
                    "lost_on_supported_dates": int((lost & ~sparse[:, None]).sum()),
                    "source": str(source),
                    "source_sha256": record["data"]["sha256"],
                }
            )
        del valid, old
    report = {
        "schema": "BRAZIL_RV_ROUND7_COVERAGE_AUDIT_V1",
        "base_store": accepted["store"],
        "fields": rows,
        "requires_source_specific_audit": missing,
        "scores_or_labels_read": False,
        "peak_rss_bytes": peak_rss_bytes(),
        "interpretation": "Physical means the causal source builder supplied a valid value. Sparse losses in rank-gauss fields are attributable to the support rule; other losses require transform/domain inspection. Counts across fields are not unique observations.",
    }
    if report["peak_rss_bytes"] > 8 * 1024**3:
        raise MemoryError("coverage audit exceeded 8 GiB")
    write_json_atomic(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    result = audit(parser.parse_args().output)
    summary = (
        pl.DataFrame(result["fields"])
        .group_by("family")
        .agg(
            pl.col(
                "physical_active_name_days",
                "lost_name_days",
                "lost_on_sparse_dates",
                "lost_on_supported_dates",
            ).sum()
        )
        .sort("family")
    )
    print(
        json.dumps(
            {
                "families": summary.to_dicts(),
                "uncovered": result["requires_source_specific_audit"],
                "peak_rss_bytes": result["peak_rss_bytes"],
            }
        )
    )


if __name__ == "__main__":
    main()
