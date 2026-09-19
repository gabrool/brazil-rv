"""Independent odd-lot availability/lag arithmetic against actual store arrays."""

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = time.perf_counter()
    pointer_path = PROJECT / "docs/v2_economic_data_scaling_run.json"
    pointer = json.loads(pointer_path.read_text())
    output = Path(pointer["root"]) / "auxiliary_tensor_audit/oddlot.json"
    if output.exists():
        raise FileExistsError("preserve completed odd-lot evidence")
    accepted = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())
    root = Path(accepted["store"]["root"])
    manifest = bound_json(
        {
            "path": str(root / "manifest.json"),
            "sha256": accepted["store"]["manifest_sha256"],
        }
    )
    sources = manifest["sources"]
    raw_binding = next(
        s for s in sources if Path(s.get("path", "")).name == "odd_lot_activity.parquet"
    )
    schedule_binding = next(
        s
        for s in sources
        if Path(s.get("path", "")).name == "b3_session_schedule_reconstructed_v1.csv"
    )
    for source in (raw_binding, schedule_binding):
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError("odd-lot input identity differs")
    schedule = pl.read_csv(schedule_binding["path"], try_parse_dates=True)
    # Keep 2009 when locating t-5. Never shift five observed rows for a security
    # across an unobserved session or discard the pre-store history first.
    calendar = schedule["trade_date"].to_numpy().astype("datetime64[D]")
    dates = np.load(root / "date_index.npy")
    if str(dates[-1]) > "2024-12-31":
        raise PermissionError("development audit only")
    isins = np.load(root / "isin_index.npy").tolist()
    source = pl.read_parquet(raw_binding["path"])
    if source["available_date"].max().isoformat() > "2024-12-31":
        raise PermissionError("development odd-lot rows only")
    if source.select(
        pl.struct("source_trade_date", "isin").is_duplicated().any()
    ).item():
        raise ValueError("multiple publication vintages require a dated as-of audit")
    t = np.searchsorted(calendar, source["source_trade_date"].to_numpy())
    if not np.array_equal(calendar[t], source["source_trade_date"].to_numpy()):
        raise ValueError("source trade outside exchange calendar")
    available = source["available_date"].to_numpy()
    lag_failures = int(np.count_nonzero(calendar[t + 1] != available))
    if lag_failures:
        raise ValueError("odd-lot source is not exactly next-session available")
    regular = source["regular_volume_brl"].to_numpy()
    odd = source["odd_lot_volume_brl"].to_numpy()
    total = regular + odd
    valid = np.isfinite(total) & (regular >= 0) & (odd >= 0) & (total > 0)
    share = np.divide(odd, total, out=np.zeros_like(odd), where=valid)
    current = source.select("isin", "available_date").with_columns(
        pl.Series("__t", t), pl.Series("share", share), pl.Series("known", valid)
    )
    prior = current.select(
        "isin",
        (pl.col("__t") + 5).alias("__t"),
        pl.col("share").alias("prior_share"),
        pl.col("known").alias("prior_known"),
        pl.col("available_date").alias("prior_available"),
    )
    joined = (
        current.join(prior, on=["isin", "__t"], how="left")
        .join(
            pl.DataFrame(
                {
                    "available_date": dates.astype(object).tolist(),
                    "__d": np.arange(len(dates)),
                }
            ),
            on="available_date",
        )
        .join(pl.DataFrame({"isin": isins, "__n": np.arange(len(isins))}), on="isin")
    )
    d, n = joined["__d"].to_numpy(), joined["__n"].to_numpy()
    active = np.load(root / "active.npy", mmap_mode="r")
    prior_known = (
        joined["prior_known"].fill_null(False).to_numpy()
        & (joined["prior_available"] <= joined["available_date"])
        .fill_null(False)
        .to_numpy()
    )
    share = joined["share"].to_numpy()
    known = joined["known"].to_numpy()
    raw_fields = [share, share - joined["prior_share"].fill_null(0).to_numpy()]
    masks = [known, known & prior_known]
    names = manifest["feature_names"]["sidecar_oddlot"]
    specs = {
        s["name"]: s
        for s in manifest["metadata"]["feature_schema"]["specifications"]
        if s["family"] == "sidecar_oddlot"
    }
    report = {
        "schema": "BRAZIL_RV_ODDLOT_TENSOR_AUDIT_V1",
        "store": accepted["store"],
        "source": raw_binding,
        "schedule": schedule_binding,
        "source_rows": source.height,
        "aligned_rows": joined.height,
        "source_start": str(source["source_trade_date"].min()),
        "source_end": str(source["source_trade_date"].max()),
        "D_plus_1_mismatches": lag_failures,
        "fields": [],
        "model_predictions_labels_or_heldout_read": False,
        "method": "independent BRL odd/(regular+odd), exact exchange-session lag-five join and prior-publication check; no production sidecar code",
        "limitations": "Bound archive has dates but no intraday publication timestamps; original COTAHIST historical revision share remains unknown. This is source-derived-volume-to-tensor evidence, not a new raw-file census.",
        "reproducer": {
            "path": str(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
    }
    for f, name in enumerate(names):
        payload = raw_fields[f].astype(np.float32)
        mask = masks[f] & active[d, n]
        if specs[name]["transform"] == "bounded_fraction":
            mask &= (payload >= 0) & (payload <= 1)
            transformed = (2 * payload.astype(float) - 1).astype(np.float32)
        elif specs[name]["transform"] == "precomputed_native":
            transformed = payload
        else:
            raise ValueError("unreviewed odd-lot transform")
        expected = np.zeros(active.shape, np.float32)
        expected[d[mask], n[mask]] = transformed[mask]
        expected_mask = np.zeros(active.shape, bool)
        expected_mask[d, n] = mask
        expected_age = np.full(active.shape, -1, np.float32)
        # Source age remains one even when an exact lag-five endpoint is absent.
        eligible = active[d, n]
        expected_age[d[eligible], n[eligible]] = 1
        counts = {}
        for suffix, desired in (
            ("values", expected),
            ("valid", expected_mask),
            ("age_sessions", expected_age),
        ):
            actual = np.load(root / f"sidecar_oddlot_{suffix}.npy", mmap_mode="r")[
                ..., f
            ]
            counts[suffix + "_mismatches"] = int(np.count_nonzero(actual != desired))
        report["fields"].append(
            {
                "name": name,
                "valid_active_cells": int(mask.sum()),
                "first_store_session_valid": int(expected_mask[0].sum()),
                **counts,
            }
        )
    report["seconds"] = time.perf_counter() - started
    report["status"] = (
        "exact"
        if not any(
            v
            for f in report["fields"]
            for k, v in f.items()
            if k.endswith("mismatches")
        )
        else "diagnose"
    )
    digest = write_json_atomic(output, report)
    pointer["oddlot_tensor_audit"] = {"path": str(output), "sha256": digest}
    write_json_atomic(pointer_path, pointer)
    print(
        json.dumps(
            {
                "status": report["status"],
                "seconds": report["seconds"],
                "fields": report["fields"],
            }
        )
    )


if __name__ == "__main__":
    main()
