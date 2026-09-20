"""Assemble verified amendments into a separate complete development store."""

import copy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.feature_spec import FeatureSpec, transform_feature_panel_into
from brazil_rv.v2.round5_store import align_family
from brazil_rv.v2.store import StoreStaging, close_memmap

PROJECT = Path(__file__).resolve().parents[1]


def verified(record):
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Amendment differs: {path}")
    return path


def inputs(contract):
    manifests = {k: bound_json(v) for k, v in contract["amendments"].items()}
    h = manifests["rename_history_propagation"]
    replacement = {**h["arrays"], **manifests["rename_peer_propagation"]["arrays"]}
    replacement.pop("date_index")
    replacement.pop("isin_index")
    issuer = manifests["rename_issuer_propagation"]["artifacts"]
    activity = manifests["activity_magnitude_propagation"]["artifacts"]
    families = {
        "fundamentals": issuer["fundamentals"],
        "events": issuer["events"],
        "sector": manifests["rename_dependents"]["artifacts"]["sector"],
        "magnitudes": activity["magnitudes"],
        "cross_market": activity["cross_market"],
        "lending": manifests["lending_feature_propagation"]["artifacts"]["features"],
    }
    # Ordered composition: native scale tail and clock normalization supersede
    # their earlier source layers, while every sparse index stays on full axes.
    patch_sources = [
        ("rename_m1_propagation", "native_deltas"),
        ("rename_m1_input_audit", "tail_deltas"),
        ("m1_scalar_assembly", "deltas"),
        ("to_close_clock", "deltas"),
        ("aeri_wealth_qualification", "deltas"),
        ("remaining_auxiliary_input_audit", "deltas"),
        ("corporate_target_propagation", "deltas"),
        ("corporate_claim_rows", "deltas"),
    ]
    patches = {}
    for label, key in patch_sources:
        with np.load(verified(manifests[label][key])) as z:
            for member in z.files:
                if not member.endswith("__indices"):
                    continue
                stem = member.removesuffix("__indices")
                name = stem.split("__")[-1]
                indices, values = z[member].copy(), z[stem + "__values"].copy()
                if len(indices):
                    patches.setdefault(name, []).append((label, indices, values))
    return manifests, replacement, families, patches


def main():
    start = perf_counter()
    contract_path = PROJECT / "docs/v2_derived_store_contract.json"
    contract = json.loads(contract_path.read_text())
    parent = contract["parent"]
    source = Path(parent["root"])
    original = bound_json(
        {"path": str(source / "manifest.json"), "sha256": parent["manifest_sha256"]}
    )
    assert original["axes"]["date_end"] <= "2024-12-30"
    reports, replacement, families, patches = inputs(contract)
    destination = Path(contract["output"])
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("Derived output overlaps immutable parent")
    evidence = Path(contract["evidence_root"])
    evidence.mkdir(exist_ok=False)
    (evidence / "executed_reproducer.py").write_bytes(Path(__file__).read_bytes())
    dates = np.load(source / "date_index.npy")
    isins = tuple(np.load(source / "isin_index.npy").tolist())
    days = dates.astype(object).tolist()
    active = np.load(verified(replacement["active"]), mmap_mode="r")
    specifications = original["metadata"]["feature_schema"]["specifications"]
    generated = {
        f"sidecar_{f}_{s}"
        for f in families
        for s in ("values", "valid", "age_sessions")
    }
    effects, layer_receipts = {}, {}
    goll = reports["goll_activity_admission"]
    with np.load(verified(goll["artifacts"]["slow_patch"])) as z:
        goll_patch = {k: z[k].copy() for k in z.files}

    def compare(name, new):
        old_record = original["arrays"].get(name)
        if old_record is None:
            return {"new_array": True, "cells": new.size}
        old = np.load(source / old_record["path"], mmap_mode="r")
        changed, gained, lost = 0, 0, 0
        for t in range(0, len(dates), 64):
            a, b = old[t : t + 64], new[t : t + 64]
            equal = a == b
            if np.issubdtype(a.dtype, np.floating):
                equal |= np.isnan(a) & np.isnan(b)
            changed += int((~equal).sum())
            if a.dtype == bool:
                gained += int((b & ~a).sum())
                lost += int((a & ~b).sum())
        close_memmap(old)
        return {
            "cells": new.size,
            "changed": changed,
            "true_gained": gained,
            "true_lost": lost,
        }

    with StoreStaging(destination, dates=dates, isins=isins) as staging:
        for name in sorted(set(original["arrays"]) | set(replacement)):
            if name in generated:
                continue
            record = replacement.get(name)
            path = (
                verified(record)
                if record
                else source / original["arrays"][name]["path"]
            )
            staging.copy_array(name, path)
            a = staging.open_array(name, mode="r+")
            layers = []
            if record:
                layers.append({"replacement": record})
            if name == "slow_values":
                ix = tuple(
                    goll_patch[k] for k in ("date_index", "name_index", "field_index")
                )
                np.testing.assert_array_equal(a[ix], goll_patch["before"])
                a[ix] = goll_patch["values"]
                layers.append({"goll_activity": len(goll_patch["values"])})
            if name in ("volume_brl", "quantity", "trade_count", "trade_observed"):
                k = {
                    "trade_count": "trade_count",
                    "quantity": "quantity",
                    "volume_brl": "volume_brl",
                    "trade_observed": "trade_observed",
                }[name]
                a[goll["date_index"], goll["name_index"]] = goll["printed_activity"][k]
                layers.append({"goll_printed_activity": True})
            for label, indices, values in patches.get(name, []):
                if indices.ndim == 1:
                    a.reshape(-1)[indices] = values
                else:
                    a[tuple(indices.T)] = values
                layers.append({"amendment": label, "cells": len(indices)})
            effects[name] = compare(name, a)
            layer_receipts[name] = layers
            close_memmap(a)
        for family, record in families.items():
            name = "sidecar_" + family
            fields = tuple(original["feature_names"][name])
            specs = [FeatureSpec(**s) for s in specifications if s["family"] == name]
            frame = pl.read_parquet(verified(record))
            outputs = [
                staging.create_array(
                    name + "_" + suffix, (*active.shape, len(fields)), dtype
                )
                for suffix, dtype in (
                    ("values", np.float32),
                    ("valid", bool),
                    ("age_sessions", np.float32),
                )
            ]
            # Cross-sectional transforms are decision-local. Date chunks retain
            # every name and need no history refit or repeated source reducer.
            for t in range(0, len(dates), 64):
                end = min(t + 64, len(dates))
                bounded = frame.filter(
                    pl.col("date").is_between(days[t], days[end - 1])
                )
                raw, mask, age = align_family(bounded, days[t:end], isins, fields)
                transform_feature_panel_into(
                    raw,
                    mask,
                    active[t:end],
                    specs,
                    outputs[0][t:end],
                    outputs[1][t:end],
                )
                outputs[2][t:end] = np.where(active[t:end, :, None], age, -1)
            for suffix, a in zip(("values", "valid", "age_sessions"), outputs):
                key = name + "_" + suffix
                effects[key] = compare(key, a)
                layer_receipts[key] = [{"family_source": record}]
                close_memmap(a)
            del frame, outputs, raw, mask, age
            print(json.dumps({"family_assembled": family}), flush=True)
        tables = {k: source / v["path"] for k, v in original["tables"].items()}
        # Prior diagnostic tables remain recoverable, explicitly labelled as
        # parent receipts rather than summaries of the new model coordinates.
        stale = [
            k
            for k in tables
            if any(
                s in k
                for s in ("coverage", "validity", "universe_size", "target_changes")
            )
        ]
        for k in stale:
            tables["parent_" + k] = tables.pop(k)
        h = reports["rename_history_propagation"]
        tables["isin_succession_links"] = verified(h["links"])
        tables["slow_history_links"] = verified(h["history_mapping"])
        tables["issuer_identity"] = verified(
            reports["rename_issuer_propagation"]["artifacts"]["identity"]
        )
        tables["corporate_claim_close_rows"] = verified(
            reports["corporate_claim_rows"]["claims"]
        )
        tables["corporate_source_entry_barriers"] = pl.DataFrame(
            reports["corporate_claim_rows"]["barriers"]
        )
        tables["m1_diagnostic_replacement"] = verified(
            reports["aeri_wealth_qualification"]["m1_diagnostic_replacement"]
        )
        tables["universe_size"] = pl.DataFrame(
            {"date": days, "active_names": active.sum(axis=1)}
        )
        tables["economic_array_changes"] = pl.DataFrame(
            [{"array": k, **v} for k, v in effects.items()]
        )
        metadata = copy.deepcopy(original["metadata"])
        metadata["economic_data_contract"] = {
            "contract": binding(contract_path),
            "parent": parent,
            "requires_refit": True,
            "old_checkpoint_coordinates_compatible": False,
            "inherited_metadata_and_parent_tables": "Historical parent receipts; amendments and current array inventory supersede their counts/source scope.",
            "claim_marks": "Audit-only closing entitlements, no source OHLC or new model feature.",
            "accounting": "Separate corporate/borrow/cash manifest required; no account return or CDI tensor replacement in this feature contract.",
            "unresolved_source_limits": contract["source_limits"],
        }
        metadata["isin_succession_link_count"] = pl.read_parquet(
            tables["isin_succession_links"]
        ).height
        staging.seal(
            feature_names=original["feature_names"],
            metadata=metadata,
            tables=tables,
            sources=[
                *original["sources"],
                binding(contract_path),
                *contract["amendments"].values(),
            ],
            maximum_peak_rss_bytes=8 * 1024**3,
        )
    result = {
        "status": "assembled_pending_combined_consumer_acceptance",
        "store": {
            "root": str(destination),
            "manifest_sha256": sha256_file(destination / "manifest.json"),
        },
        "contract": binding(contract_path),
        "parent": parent,
        "code": binding(evidence / "executed_reproducer.py"),
        "effects": effects,
        "layers": layer_receipts,
        "seconds": perf_counter() - start,
        "forecast_scoring": False,
        "heldout_access": False,
    }
    write_json_atomic(evidence / "manifest.json", result)
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    run["derived_store_assembly"] = binding(evidence / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps({"store": result["store"], "seconds": result["seconds"]}), flush=True
    )


if __name__ == "__main__":
    main()
