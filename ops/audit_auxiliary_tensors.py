"""Independently reconcile bound auxiliary rows to sealed development tensors.

This checks the last source-to-store boundary. It does not certify the earlier
publication clock merely because a producer supplied a decision date or age.
No production alignment/transform function is used by the row arithmetic oracle.
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import bound_json, source_families

PROJECT = Path(__file__).resolve().parents[1]


def binding(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def transformed(raw, known, spec):
    """Independent scalar formulas from the sealed typed feature contract."""
    values = raw.astype(np.float64)
    mask = known.copy()
    kind = spec["transform"]
    if kind == "binary":
        mask &= (values == 0) | (values == 1)
    elif kind == "bounded_fraction":
        mask &= (values >= 0) & (values <= 1)
        values = 2 * values - 1
    elif kind == "age_sessions":
        mask &= values >= 0
        log_age = np.log1p(np.maximum(values, 0))
        values = log_age / (log_age + np.log1p(252.0))
    elif kind == "annual_rate":
        values = np.arcsinh(values / 0.01)
    elif kind != "precomputed_native":
        raise ValueError(f"unreviewed auxiliary transform: {kind}")
    return np.where(mask, values, 0).astype(np.float32), mask


def reconcile_family(record, family, names, specs, root, dates, isins, active):
    source = Path(record["data"]["path"])
    if sha256_file(source) != record["data"]["sha256"]:
        raise ValueError(f"bound {family} source differs")
    keys = pl.read_parquet(source, columns=["date", "isin"])
    if keys["date"].max().isoformat() > "2024-12-31":
        raise PermissionError("auxiliary audit may not consume held-out rows")
    if keys.select(pl.struct("date", "isin").is_duplicated().any()).item():
        raise ValueError("ambiguous source row assignment")
    axis = (
        keys.with_row_index("__row")
        .join(pl.DataFrame({"date": dates, "__d": np.arange(len(dates))}), on="date")
        .join(pl.DataFrame({"isin": isins, "__n": np.arange(len(isins))}), on="isin")
        .sort("__row")
    )
    if axis.height != keys.height:
        raise ValueError("unmapped dated security source rows")
    d, n = axis["__d"].to_numpy(), axis["__n"].to_numpy()
    schema = pl.read_parquet_schema(source)
    arrays = {
        suffix: np.load(root / f"{family}_{suffix}.npy", mmap_mode="r")
        for suffix in ("values", "valid", "age_sessions")
    }
    fields = []
    for f, name in enumerate(names):
        columns = [
            c for c in (name, name + "_mask", name + "_age_sessions") if c in schema
        ]
        frame = pl.read_parquet(source, columns=columns) if columns else None
        raw = (
            frame[name].cast(pl.Float32).to_numpy()
            if name in columns
            else np.zeros(keys.height, dtype=np.float32)
        )
        known = np.isfinite(raw) & (name in columns)
        if name + "_mask" in columns:
            known &= frame[name + "_mask"].fill_null(False).to_numpy()
        age = (
            frame[name + "_age_sessions"].cast(pl.Float32).to_numpy()
            if name + "_age_sessions" in columns
            else np.full(keys.height, -1, np.float32)
        )
        if np.any(known & (~np.isfinite(age) | (age < 0))):
            raise ValueError(f"known source field has no age: {family}/{name}")
        value, valid = transformed(raw, known & active[d, n], specs[name])
        expected = np.zeros(active.shape, dtype=np.float32)
        expected[d, n] = value
        expected_valid = np.zeros(active.shape, dtype=bool)
        expected_valid[d, n] = valid
        expected_age = np.full(active.shape, -1, dtype=np.float32)
        has_age = np.isfinite(age) & (age >= 0) & active[d, n]
        expected_age[d[has_age], n[has_age]] = age[has_age]
        counts = {
            "value_mismatches": int(
                np.count_nonzero(expected != arrays["values"][..., f])
            ),
            "mask_mismatches": int(
                np.count_nonzero(expected_valid != arrays["valid"][..., f])
            ),
            "age_mismatches": int(
                np.count_nonzero(expected_age != arrays["age_sessions"][..., f])
            ),
        }
        fields.append(
            {
                "field": name,
                "transform": specs[name]["transform"],
                "source_units": specs[name]["source_units"],
                "valid_active_cells": int(valid.sum()),
                "source_known_inactive_cells": int(
                    np.count_nonzero(known & ~active[d, n])
                ),
                "domain_rejections": int(
                    np.count_nonzero(known & active[d, n] & ~valid)
                ),
                "maximum_known_age_sessions": float(age[valid].max())
                if valid.any()
                else None,
                **counts,
            }
        )
    upstream = {}
    for key in ("source_manifest", "availability_proof"):
        if key in record:
            evidence = bound_json(record[key])
            upstream[key] = {**record[key], "keys": list(evidence)}
    return {
        "family": family,
        "data": record["data"],
        "rows": keys.height,
        "row_assignment": "one-to-one exact decision date and ISIN; no ticker join or fill",
        "original_publication_columns": [
            c
            for c in schema
            if any(
                word in c
                for word in ("timestamp", "public_available", "receipt", "delivery")
            )
        ],
        "upstream": upstream,
        "fields": fields,
        "mismatches": {
            key: sum(row[key] for row in fields)
            for key in ("value_mismatches", "mask_mismatches", "age_mismatches")
        },
    }


def main():
    started = time.perf_counter()
    pointer_path = PROJECT / "docs/v2_economic_data_scaling_run.json"
    pointer = json.loads(pointer_path.read_text())
    output = Path(pointer["root"]) / "auxiliary_tensor_audit"
    output.mkdir(exist_ok=True)
    report_path = output / "report.json"
    if report_path.exists():
        raise FileExistsError(
            "preserve completed audit; diagnose its findings before repeating"
        )
    accepted = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())
    root = Path(accepted["store"]["root"])
    manifest = bound_json(
        {
            "path": str(root / "manifest.json"),
            "sha256": accepted["store"]["manifest_sha256"],
        }
    )
    if manifest["axes"]["date_end"] > "2024-12-31":
        raise PermissionError("audit is development only")
    parent = manifest["metadata"]["data_repair"]["parent"]
    sources = source_families(
        bound_json(
            {
                "path": str(Path(parent["root"]) / "manifest.json"),
                "sha256": parent["manifest_sha256"],
            }
        )
    )
    financial = bound_json(accepted["financial_family"])
    sources["fundamentals"] = {
        "data": financial["data"],
        "source_manifest": accepted["financial_family"],
    }
    dates = np.load(root / "date_index.npy").astype(object).tolist()
    isins = np.load(root / "isin_index.npy").tolist()
    active = np.load(root / "active.npy", mmap_mode="r")
    report = {
        "schema": "BRAZIL_RV_AUXILIARY_TENSOR_AUDIT_V1",
        "store": accepted["store"],
        "reproducer": binding(Path(__file__)),
        "families": [],
        "model_predictions_labels_or_heldout_read": False,
        "immutable_store_modified": False,
        "scope": "nine decision-dated producer parquets through every actual tensor cell; independent transform and alignment oracle",
        "limitations": [
            "Producer decision dates/ages are not original publication evidence; upstream receipt-to-decision clocks still require review",
            "Oddlot requires its distinct publication/vintage and 2009 lag-five warmup reconstruction; not covered by this nine-family boundary",
            "No fit scaler, final neural tensor, source revision or forecasting acceptance follows from this last-mile check",
        ],
    }
    specifications = manifest["metadata"]["feature_schema"]["specifications"]
    for family, names in sorted(manifest["feature_names"].items()):
        if not family.startswith("sidecar_") or family == "sidecar_oddlot":
            continue
        specs = {s["name"]: s for s in specifications if s["family"] == family}
        result = reconcile_family(
            sources[family.removeprefix("sidecar_")],
            family,
            names,
            specs,
            root,
            dates,
            isins,
            active,
        )
        report["families"].append(result)
        write_json_atomic(output / "progress.json", report)
        print(json.dumps({"family": family, **result["mismatches"]}), flush=True)
    counts = Counter()
    for result in report["families"]:
        counts.update(result["mismatches"])
    report["mismatches"] = dict(counts)
    report["fields"] = sum(len(r["fields"]) for r in report["families"])
    report["valid_active_cells"] = sum(
        f["valid_active_cells"] for r in report["families"] for f in r["fields"]
    )
    report["seconds"] = time.perf_counter() - started
    report["status"] = (
        "last_boundary_exact_upstream_audit_pending"
        if not any(counts.values())
        else "mismatches_require_diagnosis"
    )
    write_json_atomic(report_path, report)
    pointer["auxiliary_tensor_audit"] = binding(report_path)
    write_json_atomic(pointer_path, pointer)
    print(json.dumps({"status": report["status"], "seconds": report["seconds"]}))


if __name__ == "__main__":
    main()
