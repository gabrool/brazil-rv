"""Independent dated free-float selection and utilization arithmetic."""

from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import date
import json
from pathlib import Path
import time

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = time.perf_counter()
    run = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    output = Path(run["root"]) / "lending_denominator_audit"
    output.mkdir(exist_ok=False)
    peers = bound_json(run["fca_peer_propagation"])
    identity_source = bound_json(run["fca_identity_admission"])["artifacts"]["identity"]
    inputs = bound_json(peers["lending_source_bindings"])

    def parquet(source):
        assert sha256_file(Path(source["path"])) == source["sha256"]
        return pl.read_parquet(source["path"])

    identity = parquet(identity_source)
    balances = parquet(inputs["old_lending"]["balances"])
    floats = parquet(inputs["new_cvm_files"]["free_float_observations.parquet"])
    changes = bound_json(inputs["new_cvm_files"]["capital_change_observations.json"])
    market = Path(inputs["base_inputs"][0]["path"]).parent
    for receipt in inputs["base_inputs"]:
        assert sha256_file(Path(receipt["path"])) == receipt["sha256"]
    days = np.load(market / "date_index.npy").astype("datetime64[D]").tolist()
    day_index = {d: i for i, d in enumerate(days)}
    columns = {s: i for i, s in enumerate(np.load(market / "isin_index.npy"))}
    observed = np.load(market / "observed.npy", mmap_mode="r")
    dist = np.load(market / "distribution_number.npy", mmap_mode="r")
    barriers = np.array(
        np.load(market / "detected_split_mask.npy", mmap_mode="r")
        | np.load(market / "ambiguous_action_mask.npy", mmap_mode="r")
    )
    for col in range(observed.shape[1]):
        rows = np.flatnonzero(observed[:, col] & np.isfinite(dist[:, col]))
        barriers[rows[1:][np.diff(dist[rows, col]) != 0], col] = True
    boundaries = [np.flatnonzero(barriers[:, col]) for col in range(barriers.shape[1])]
    identity = identity.with_columns(
        pl.col("cnpj").str.slice(0, 8).alias("issuer_root")
    )
    counts = identity.group_by("date", "issuer_root", "cvm_code", "class").agg(
        pl.col("isin").n_unique().alias("members")
    )
    identity = identity.join(counts, on=["date", "issuer_root", "cvm_code", "class"])
    joined = (
        balances.with_columns(
            pl.col("available_date").alias("date"),
            pl.col("security_id").str.strip_prefix("ISIN:").alias("isin"),
        )
        .join(identity, on=["date", "isin"], how="left")
        .sort("date", "isin")
    )
    float_groups = defaultdict(list)
    for row in floats.iter_rows(named=True):
        if row["snapshot_date"] is not None and row["free_float_shares"] > 0:
            float_groups[row["cnpj"][:8], row["cvm_code"], row["class"]].append(row)
    event_groups = defaultdict(list)
    for e in changes:
        event_groups[e["cnpj"][:8], e["cvm_code"]].append(
            dict(e, effective=date.fromisoformat(e["effective"]))
        )
    reasons = Counter()
    selected = []
    out = []
    future_excluded = 0
    for row in joined.iter_rows(named=True):
        index = day_index[row["date"]]
        if (
            row["class"] not in ("ON", "PN")
            or row["members"] != 1
            or row["identity_effective_start"] > row["source_position_date"]
        ):
            reasons["identity_or_class"] += 1
            continue
        key = (row["issuer_root"], row["cvm_code"], row["class"])
        published = [d for d in float_groups[key] if d["date"] <= row["date"]]
        candidates = [
            d for d in published if d["snapshot_date"] <= row["source_position_date"]
        ]
        future_excluded += sum(
            d["snapshot_date"] > row["source_position_date"] for d in published
        )
        if not candidates:
            reasons["no_dated_float"] += 1
            continue
        d = max(
            candidates,
            key=lambda d: (
                d["snapshot_date"],
                d["reference"],
                d["version"],
                d["date"],
                d["document_id"],
            ),
        )
        if any(
            e["available_index"] <= index
            and d["snapshot_date"] < e["effective"] <= row["source_position_date"]
            for e in event_groups[key[:2]]
        ):
            reasons["known_capital_event"] += 1
            continue
        col = columns.get(row["isin"])
        last = day_index.get(row["source_position_date"])
        first = bisect_right(days, d["snapshot_date"])
        if (
            col is None
            or last is None
            or first > last + 1
            or np.any((boundaries[col] >= first) & (boundaries[col] <= last))
        ):
            reasons["observed_unit_boundary"] += 1
            continue
        if (
            row["lending_balance_quantity"] is None
            or row["lending_balance_quantity"] < 0
        ):
            reasons["missing_quantity"] += 1
            continue
        out.append(
            {
                "date": row["date"],
                "isin": row["isin"],
                "utilization_proxy": row["lending_balance_quantity"]
                / d["free_float_shares"],
                "utilization_proxy_age_sessions": index
                - min(last, day_index[d["date"]]),
            }
        )
        selected.append(
            {
                "date": row["date"],
                "isin": row["isin"],
                "position_date": row["source_position_date"],
                "document_id": d["document_id"],
                "float_known_date": d["date"],
                "snapshot_date": d["snapshot_date"],
                "quantity": row["lending_balance_quantity"],
                "denominator": d["free_float_shares"],
            }
        )
    actual = (
        parquet(peers["artifacts"]["lending"])
        .select("date", "isin", "utilization_proxy", "utilization_proxy_age_sessions")
        .drop_nulls("utilization_proxy")
    )
    # Family storage is Float32; the ratio is formed before the explicit cast.
    expected = pl.DataFrame(out).cast(actual.schema)
    assert_frame_equal(
        actual.sort("date", "isin"), expected.sort("date", "isin"), check_exact=True
    )
    selected = pl.DataFrame(selected)
    selected.write_parquet(output / "selected_denominators.parquet")
    report = {
        "schema": "LENDING_DENOMINATOR_AUDIT_V1",
        "identity": identity_source,
        "peer_propagation": run["fca_peer_propagation"],
        "source_bindings": peers["lending_source_bindings"],
        "rows": len(joined),
        "valid_rows": len(expected),
        "rejected": dict(reasons),
        "future_measurements_excluded": future_excluded,
        "selected_small_denominators_below_10000": selected.filter(
            pl.col("denominator") < 10000
        ).height,
        "mismatches": 0,
        "seconds": time.perf_counter() - started,
        "reproducer": binding(Path(__file__)),
        "artifacts": {
            "selected_denominators": binding(output / "selected_denominators.parquet")
        },
        "limits": [
            "Matches identity-only propagated lending on its original balances and market-boundary contract; recovered balance/rate additions still require separate feature propagation.",
            "Public float and loan balances do not establish executable locate capacity. Single-vintage source revisions remain unknown.",
            "No original source, accepted store or forecast was modified.",
        ],
    }
    write_json_atomic(output / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
