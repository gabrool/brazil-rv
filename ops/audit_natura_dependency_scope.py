"""Bind unchanged Natura dependencies using actual producer inputs and assignments."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    daily = bound_json(run["natura_daily_propagation"])
    plan = bound_json(daily["plan"])
    parent = plan["parent_store"]
    root = Path(parent["root"])
    m = bound_json(
        {"path": str(root / "manifest.json"), "sha256": parent["manifest_sha256"]}
    )
    dates, isins = np.load(root / "date_index.npy"), np.load(root / "isin_index.npy")
    n = int(np.flatnonzero(isins == "BRNATUACNOR6")[0])
    start, _, stop = plan["rows"]
    window = slice(start, stop)
    old_m1 = bound_json(run["rename_m1_propagation"])
    assignments = pl.read_parquet(old_m1["assignments"]["path"]).filter(
        pl.col("isin") == isins[n]
    )
    assert assignments.height == 1
    a = assignments.row(0, named=True)
    assert str(a["first_overlap_date"]) > str(dates[-1])
    mapping = pl.read_parquet(
        root / m["tables"]["native_fast_security_mapping"]["path"]
    )
    f = mapping.filter(pl.col("store_name_index") == n)["fast_index"][0]
    tests = []
    for key, axis in (
        ("fast_present", n),
        ("fast_patch_valid", f),
        ("fast_patch_mask", f),
        ("intraday_valid", n),
        ("target_to_close_valid", n),
    ):
        values = np.load(root / m["arrays"][key]["path"], mmap_mode="r")[window, axis]
        assert not values.any()
        tests.append({"array": key, "cells": values.size, "supported": 0})
    # Exact source keys used by valuation_market; no q/cash or wealth input.
    unchanged = [
        "active",
        "observed",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "volume_brl",
        "quantity",
        "trade_count",
        "activity_valid",
        "distribution_number",
        "distribution_change_mask",
        "detected_split_mask",
        "ambiguous_action_mask",
        "action_has_action",
        "action_session_resolved",
        "action_successor_index",
    ]
    out = Path(daily["plan"]["path"]).parent / "dependency_scope"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    assignments.write_parquet(out / "natura_assignment.parquet")
    sources = [
        "round5_cvm.py",
        "round5_b3.py",
        "round5_exposures.py",
        "round5_magnitude.py",
        "round5_index.py",
        "auxiliary_history.py",
        "intraday_features.py",
    ]
    code = {}
    for name in sources:
        path = PROJECT / "research/src/brazil_rv/v2" / name
        if path.exists():
            (out / name).write_bytes(path.read_bytes())
            code[name] = binding(out / name)
    report = {
        "status": "passed_assignment_and_dependency_scope_not_final_combined_store",
        "parent_store": parent,
        "daily_plan": daily["plan"],
        "no_m1_observations_read": True,
        "assignment": binding(out / "natura_assignment.parquet"),
        "native_and_scalar_checks": tests,
        "no_native_scalar_or_to_close_amendment": "The sole accepted NATU assignment begins 2025-07-02. No admitted 2019 observations exist. All own-name native/scalar/to-close support is empty over the entire bounded source window. A changed own-name daily sigma cannot affect other names' native channels or absent scalar cross-section entries.",
        "unchanged_inputs": {
            key: {**m["arrays"][key], "path": str(root / m["arrays"][key]["path"])}
            for key in unchanged
        },
        "families": {
            "fundamentals": "Issuer filings, selected accounts, literal share counts, dated identity and raw prior closes unchanged. valuation_market uses DISMES/detected/ambiguous unit barriers, not inferred q/cash or shareholder wealth. No financial rebuild is warranted.",
            "events": "Existing original CVM publication-time ledger and issuer identity unchanged; these sourced notices are already historical disclosures, not newly timed events.",
            "lending": "Rates, balances, raw activity/ADV and issuer/float observations unchanged. The same valuation_market unit barrier enters utilization. No rate/balance/loan alias or denominator adjustment is inferred from scalar shareholder terms.",
            "microstructure_options_oddlot_rebalance": "Raw source activity, cash/option prices and quantities, publication dates, portfolio composition, raw ADV, identity and membership unchanged. These reducers do not consume shareholder wealth, target sigma, beta or monthly peer labels.",
            "sector_magnitudes_cross_market": "Wealth-dependent returns, physical volatility/beta and issuer-shrunk exposures require the separate bounded propagation. Macro shocks, feature-CDI coordinates, flow/volume interactions and ADR source contracts stay fixed.",
        },
        "barrier_limit": "Corrected contractual q/cash does not retrospectively rewrite the pre-existing market detector or DISMES barrier. That diagnostic is not proof of a split. Removing conservative feature support requires a separate explicit feature contract, not an unlabelled tax/share-count inference.",
        "account_limit": "Sep20 bonus delivery and Feb26 JCP payment/withholding still require separate account treatment; none is an observed fill or cash reinvestment in this data amendment.",
        "source_code": code,
        "seconds": perf_counter() - started,
    }
    write_json_atomic(out / "manifest.json", report)
    run["natura_dependency_scope"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps({"checks": tests, "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
