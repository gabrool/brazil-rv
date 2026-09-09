"""Compare the repaired store to the canonical store on authorized common axes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from .artifacts import inventory, sha256_file, write_json_atomic
from .contract import DEVELOPMENT_END, FINETUNE_START, HORIZONS, PRETRAIN_END
from .data_foundation import continuation_identity_axis
from .store import open_store_for_samples, peak_rss_bytes


def monthly_coverage(store, dates: np.ndarray) -> list[dict[str, object]]:
    """Feature and family coverage on the causal active universe, month by month."""
    months = dates.astype("datetime64[M]")
    rows = []
    for family, features in store.manifest["feature_names"].items():
        native = family == "native_fast"
        date_level = family == "common_state_diagnostic"
        key = "fast_patch_valid" if native else f"{family}_valid"
        if key not in store.manifest["arrays"]:
            continue
        if native:
            native_indices = (
                store.read_table("native_fast_security_mapping")
                .sort("fast_index")
                .get_column("store_name_index")
                .to_numpy()
            )
        for month in np.unique(months):
            indices = np.flatnonzero(months == month)
            active = store.read("active", indices).astype(bool)
            valid = store.read(key, indices).astype(bool)
            if native:
                valid = valid.any(axis=2) & active[:, native_indices, None]
            elif not date_level:
                valid &= active[..., None]
            denominator = len(indices) if date_level else int(active.sum())
            counts = valid.sum(axis=0 if date_level else (0, 1))
            rows.append(
                {
                    "family": family,
                    "month": str(month),
                    "sessions": len(indices),
                    "active_name_days": int(active.sum()),
                    "possible_observations": denominator,
                    "feature_count": len(features),
                    "coverage_unit": "session_with_valid_diagnostic"
                    if date_level
                    else "active_name_day_with_any_valid_patch"
                    if native
                    else "active_name_day_with_valid_feature",
                    **(
                        {
                            "mapped_active_name_days": int(
                                active[:, native_indices].sum()
                            ),
                            "fast_present_active_name_days": int(
                                (store.read("fast_present", indices) & active).sum()
                            ),
                        }
                        if native
                        else {}
                    ),
                    "any_feature_valid_observations": int(valid.any(axis=-1).sum()),
                    "mean_feature_coverage": float(
                        counts.sum() / (denominator * len(features))
                    )
                    if denominator and features
                    else None,
                    "features": {
                        name: {
                            "valid_observations": int(count),
                            "fraction": float(count / denominator)
                            if denominator
                            else None,
                        }
                        for name, count in zip(features, counts, strict=True)
                    },
                }
            )
    return rows


def registered_change(name: str) -> bool:
    return name.startswith(
        (
            "intraday_",
            "m1_cotahist_",
            "target_to_close",
            "sidecar_lending_",
            "sidecar_oddlot_",
        )
    ) or name in {"fast_sigma", "audit_eventual_survives_to_final_year"}


def _survival_audit(store, dates: np.ndarray) -> dict[str, object] | None:
    name = "audit_eventual_survives_to_final_year"
    if name not in store.manifest["arrays"]:
        return None
    # This hashed audit table is intentionally outside the model table API.
    links = pl.read_parquet(
        store.root / store.manifest["tables"]["isin_succession_links"]["path"]
    )
    identities = np.asarray(continuation_identity_axis(store.isins, links))
    final_year = dates[-1].astype("datetime64[Y]")
    final_rows = np.flatnonzero(dates.astype("datetime64[Y]") == final_year)
    observed = store.read("observed", final_rows).any(axis=0)
    expected = np.isin(identities, identities[observed])
    actual = store.read(name, np.arange(len(dates)))
    differences = int((actual != expected[None, :]).sum())
    return {
        "final_year": str(final_year),
        "expected_survivor_names": int(expected.sum()),
        "different_cells": differences,
        "exact_from_final_year_observation_and_continuation_identity": differences == 0,
    }


def _open_audit_store(root, rows, dates, *, verify_hashes=True):
    # The pretrain embargo remains unsampleable. As in the beta builder, it is
    # covered only as bounded causal history of later authorized samples.
    samples = rows[
        (dates <= np.datetime64(PRETRAIN_END))
        | (dates >= np.datetime64(FINETUNE_START))
    ]
    return open_store_for_samples(
        root,
        samples,
        purpose="evaluation",
        history_lookbacks=60,
        history_end_offsets=0,
        verify_hashes=verify_hashes,
    )


def compare(*, previous: Path, current: Path, output: Path) -> dict[str, object]:
    dates = np.load(current / "date_index.npy", allow_pickle=False)
    old_dates = np.load(previous / "date_index.npy", allow_pickle=False)
    if dates[-1] != np.datetime64(DEVELOPMENT_END):
        raise ValueError("rebuilt consumer calendar must end at 2024-12-30")
    old_rows = np.searchsorted(old_dates, dates)
    if not np.array_equal(old_dates[old_rows], dates):
        raise ValueError("rebuilt calendar differs from the canonical date slice")
    rows = np.arange(len(dates), dtype=np.int64)
    old, old_access = _open_audit_store(previous, old_rows, dates)
    new, new_access = _open_audit_store(current, rows, dates)
    comparisons, failures = [], []
    try:
        if old.isins != new.isins:
            raise ValueError("rebuilt security axis differs from canonical metadata")
        survival = _survival_audit(new, dates)
        if survival and survival["different_cells"]:
            failures.append("audit_eventual_survives_to_final_year")
        # Physical arrays only: the virtual neutral target follows from the
        # unchanged shareholder returns, sigma and neutralization characteristics.
        old_names, new_names = set(old.manifest["arrays"]), set(new.manifest["arrays"])
        for number, name in enumerate(sorted(old_names | new_names)):
            if number:
                # Hashes were verified above. Reopen only to release mapped pages
                # between arrays instead of retaining two complete stores in RSS.
                old.close()
                new.close()
                old, _ = _open_audit_store(
                    previous, old_rows, dates, verify_hashes=False
                )
                new, _ = _open_audit_store(current, rows, dates, verify_hashes=False)
            changed = registered_change(name)
            record = {"array": name, "registered_change": changed}
            if name not in old_names or name not in new_names:
                record["status"] = "added" if name in new_names else "removed"
                if not changed:
                    failures.append(name)
                comparisons.append(record)
                continue
            shape = (len(rows), *new.array_shape(name)[1:])
            if old.array_shape(name)[1:] != shape[1:] or old.array_dtype(
                name
            ) != new.array_dtype(name):
                record["status"] = "shape_or_dtype_changed"
                record.update(
                    previous_shape=old.array_shape(name),
                    current_shape=new.array_shape(name),
                    previous_dtype=old.array_dtype(name).str,
                    current_dtype=new.array_dtype(name).str,
                )
                if not changed:
                    failures.append(name)
                comparisons.append(record)
                continue
            header = (
                new.array_dtype(name).str + "\0" + ",".join(map(str, shape)) + "\0"
            ).encode("ascii")
            digests = [hashlib.sha256(header), hashlib.sha256(header)]
            differences = 0
            examples = []
            for start in range(0, len(rows), 32):
                selected = rows[start : start + 32]
                left, right = (
                    old.read(name, old_rows[selected]),
                    new.read(name, selected),
                )
                if name == "target_normalized_cross_section_valid":
                    # This auxiliary target-validity table has no payload accessor.
                    # Censor its endpoint metadata under the same (t,t+H] rule.
                    allowed = selected[:, None] + np.asarray(HORIZONS)[None, :] < len(
                        rows
                    )
                    left &= allowed
                    right &= allowed
                for digest, values in zip(digests, (left, right), strict=True):
                    digest.update(np.ascontiguousarray(values).tobytes())
                bits = np.dtype((np.void, left.dtype.itemsize))
                different = left.view(bits) != right.view(bits)
                count = int(different.sum())
                differences += count
                if count and len(examples) < 5:
                    for coordinate in np.argwhere(different)[: 5 - len(examples)]:
                        examples.append(
                            {
                                "date": str(dates[selected[coordinate[0]]]),
                                "trailing_coordinate": coordinate[1:].tolist(),
                            }
                        )
            record.update(
                {
                    "shape": shape,
                    "dtype": new.array_dtype(name).str,
                    "previous_authorized_sha256": digests[0].hexdigest(),
                    "current_authorized_sha256": digests[1].hexdigest(),
                    "different_cells": differences,
                    "examples": examples,
                    "status": "exact"
                    if digests[0].digest() == digests[1].digest()
                    else "different",
                }
            )
            if record["status"] != "exact" and not changed:
                failures.append(name)
            comparisons.append(record)
            print(
                json.dumps(
                    {key: record[key] for key in ("array", "status", "different_cells")}
                ),
                flush=True,
            )
        # Releasing mappings before the coverage pass bounds audit RSS too.
        old.close()
        new.close()
        new, _ = _open_audit_store(current, rows, dates, verify_hashes=False)
        coverage = monthly_coverage(new, dates)
        build_metadata = new.manifest["metadata"]
        result = {
            "schema": "BRAZIL_RV_V2_BOUNDED_STORE_COMPARISON",
            "status": "stop" if failures else "passed",
            "unexpected_differences": failures,
            "arrays": comparisons,
            "survival_audit": survival,
            "previous": {
                "root": str(previous),
                "manifest_sha256": sha256_file(previous / "manifest.json"),
            },
            "current": {
                "root": str(current),
                "manifest_sha256": sha256_file(current / "manifest.json"),
            },
            "calendar": {
                "first": str(dates[0]),
                "last": str(dates[-1]),
                "sessions": len(rows),
                "names": len(new.isins),
            },
            "target_comparison": "authorized (t,t+H] endpoints only; unavailable target payload is not decoded",
            "pretrain_embargo_access": "bounded causal history only; never a training, selection, or evaluation sample",
            "previous_access": old_access.payload(),
            "current_access": new_access.payload(),
            "build_peak_rss_bytes": build_metadata.get("build_peak_rss_bytes"),
            "comparison_peak_rss_gib": peak_rss_bytes() / 1024**3,
            "official_validation_accessed": False,
            "test_accessed": False,
        }
    finally:
        old.close()
        new.close()
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "comparison.json", result)
    write_json_atomic(output / "monthly_feature_coverage.json", coverage)
    write_json_atomic(output / "artifact_inventory.json", inventory(output))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(previous=args.previous, current=args.current, output=args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "unexpected_differences": result["unexpected_differences"],
            }
        )
    )


if __name__ == "__main__":
    main()
