"""Check new dependency scopes and their actual dated-family alignment."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json, source_families
from brazil_rv.v2.round5_derived import verified
from brazil_rv.v2.round5_store import align_family
from propagate_fca_financials import compare

PROJECT = Path(__file__).resolve().parents[1]
KEYS = ["date", "isin"]


def main():
    started = perf_counter()
    run = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    pointer = json.loads(
        (PROJECT / "docs/v2_data_inputs.json").read_text(encoding="utf8")
    )
    store = Path(pointer["store"]["root"])
    manifest = bound_json(
        {
            "path": str(store / "manifest.json"),
            "sha256": pointer["store"]["manifest_sha256"],
        }
    )
    sources = source_families(manifest)
    rename = bound_json(run["rename_history_propagation"])
    issuer = bound_json(run["rename_issuer_propagation"])
    deps = bound_json(run["rename_dependents"])
    activity = bound_json(run["activity_magnitude_propagation"])
    prior_peer = bound_json(run["fca_peer_propagation"])
    prior_fin = bound_json(run["fca_financial_propagation"])
    before_id = pl.read_parquet(
        verified(bound_json(run["fca_identity_admission"])["artifacts"]["identity"])
    )
    after_id = pl.read_parquet(verified(issuer["artifacts"]["identity"]))
    axis = np.load(store / "date_index.npy")
    sessions = axis.astype(object).tolist()
    isins = np.load(store / "isin_index.npy").tolist()
    active = np.load(verified(rename["arrays"]["active"]), mmap_mode="r")
    ti, ni = np.where(active)
    keys = pl.DataFrame({"date": axis[ti], "isin": np.asarray(isins)[ni]})
    allowed = []
    links = pl.read_parquet(verified(rename["history_mapping"])).to_dicts()
    for link in links:
        begin = sessions[max(link["known_index"], link["effective_index"])]
        old = (
            before_id.filter(
                (pl.col("isin") == isins[link["predecessor_index"]])
                & (pl.col("date") <= begin)
            )
            .sort("date")
            .tail(1)
        )
        sector = old["sector"][0]
        allowed.append(
            pl.concat([before_id, after_id])
            .filter((pl.col("date") >= begin) & (pl.col("sector") == sector))
            .select(KEYS)
        )
    allowed = pl.concat(allowed).unique()
    renamed = {
        isins[r[k]] for r in links for k in ["predecessor_index", "successor_index"]
    }
    current = {
        "fundamentals": issuer["artifacts"]["fundamentals"],
        "events": issuer["artifacts"]["events"],
        "sector": deps["artifacts"]["sector"],
        "magnitudes": activity["artifacts"]["magnitudes"],
        "cross_market": activity["artifacts"]["cross_market"],
    }
    old_records = {
        "fundamentals": prior_fin["artifacts"]["fundamentals"],
        "events": prior_fin["artifacts"]["events"],
        "sector": prior_peer["artifacts"]["sector"],
        "magnitudes": sources["magnitudes"]["data"],
        "cross_market": prior_peer["artifacts"]["cross_market"],
    }
    sampled = np.unique(
        np.array(
            [
                np.searchsorted(axis, np.datetime64(d))
                for d in [
                    "2011-02-16",
                    "2011-02-17",
                    "2011-03-01",
                    "2011-03-18",
                    "2023-10-24",
                    "2023-10-25",
                    "2023-10-26",
                    "2023-11-01",
                    "2024-11-14",
                    "2024-11-18",
                    "2024-11-19",
                    "2024-12-30",
                ]
            ]
        )
    )
    days = [sessions[t] for t in sampled]
    index = {d: i for i, d in enumerate(days)}
    columns = {s: i for i, s in enumerate(isins)}
    results = {}
    cell_count = 0
    for name, record in current.items():
        new = pl.read_parquet(verified(record))
        old = pl.read_parquet(verified(old_records[name]))
        effects = compare(old, new, keys)
        assert not any(v["lost"] for v in effects.values()), name
        joined = old.join(new, on=KEYS, how="full", coalesce=True, suffix="_new").join(
            keys, on=KEYS
        )
        if name in ["sector", "cross_market"]:
            unaffected = joined.join(allowed, on=KEYS, how="anti").filter(
                ~pl.col("isin").is_in(renamed)
            )
        else:
            unaffected = joined.filter(
                ~pl.col("isin").is_in(renamed | {"BRGOLLACNPR4"})
            )
        for c in [c for c in old.columns if c not in KEYS]:
            a, b = unaffected[c], unaffected[c + "_new"]
            if c.endswith("_age_sessions"):
                a, b = a.fill_null(-1), b.fill_null(-1)
            assert a.eq_missing(b).all(), (
                name,
                c,
                "unexpected change outside dependency scope",
            )
        # Use original field order, including native and independently aged
        # observations, through the actual family loader; no scaler is refitted.
        frame = new.filter(pl.col("date").is_in(days))
        names = tuple(
            c
            for c in frame.columns
            if c not in KEYS
            and not c.endswith("_age_sessions")
            and c + "_age_sessions" in frame.columns
            and "oldest_dependency" not in c
        )
        value, valid, ages = align_family(frame, days, tuple(isins), names)
        expect = np.zeros_like(value)
        seen = np.zeros_like(valid)
        age = np.full_like(ages, -1)
        for row in frame.iter_rows(named=True):
            t, n = index[row["date"]], columns[row["isin"]]
            for j, c in enumerate(names):
                v, a = row[c], row[c + "_age_sessions"]
                if v is not None and np.isfinite(v):
                    expect[t, n, j] = v
                    seen[t, n, j] = True
                if a is not None and np.isfinite(a) and a >= 0:
                    age[t, n, j] = a
        np.testing.assert_array_equal(value, expect)
        np.testing.assert_array_equal(valid, seen)
        np.testing.assert_array_equal(ages, age)
        cell_count += value.size * 3
        results[name] = {
            "effects": effects,
            "outside_scope_exact": True,
            "alignment_fields": len(names),
            "aligned_cells": value.size * 3,
        }
        if name == "magnitudes":
            # The corrected Float32 activity path returns old valid ALOS log
            # magnitude exactly once the inherited 20-day window has expired.
            both = old.join(new, on=KEYS, suffix="_new").filter(
                (pl.col("isin") == "BRALOSACNOR5")
                & (pl.col("date") >= sessions[links[0]["effective_index"] + 20])
            )
            assert_frame_equal(
                both.select("log_traded_value_20"),
                both.select(
                    pl.col("log_traded_value_20_new").alias("log_traded_value_20")
                ),
                check_exact=True,
            )
    output = Path(run["root"]) / "dependency_update_audit.json"
    write_json_atomic(
        output,
        {
            "status": "passed",
            "families": results,
            "sampled_dates": [str(d) for d in days],
            "full_name_count": len(isins),
            "aligned_cells": cell_count,
            "consumer": "round5_store.align_family plus independent per-field/date/ISIN reconstruction",
            "seconds": perf_counter() - started,
            "inputs": current,
            "code": binding(Path(__file__)),
            "limits": [
                "Intermediate dated family alignment, not acceptance of the complete new neural store.",
                "Previously verified saved conditioning and old model coordinates stay frozen.",
                "Remaining M1, lending source and contractual wealth propagation still required.",
            ],
        },
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "aligned_cells": cell_count,
                "seconds": perf_counter() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
