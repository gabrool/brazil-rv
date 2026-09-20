"""Apply the already reviewed AERI no-action disposition to its last wealth row."""

import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    root = Path(run["root"]) / "aeri_boundary"
    evidence = bound_json(binding(root / "manifest.json"))
    store = Path(evidence["parent"]["root"])
    shutil.copyfile(__file__, root / "wealth_executed.py")
    day = evidence["round7_disposition"]["date_index"]
    name = evidence["round7_disposition"]["name_index"]
    dates = np.load(store / "date_index.npy")
    assert day == len(dates) - 1
    prior_price = np.round(
        float(np.load(store / "raw_close.npy", mmap_mode="r")[day - 1, name]), 2
    )
    prior_wealth = float(
        np.load(store / "shareholder_wealth_close.npy", mmap_mode="r")[day - 1, name]
    )
    scale = prior_wealth / prior_price
    q = evidence["round7_disposition"]["old_q"]
    changes, effects = {}, {}
    for price in ("open", "high", "low", "close"):
        raw = np.round(
            float(np.load(store / ("raw_" + price + ".npy"), mmap_mode="r")[day, name]),
            2,
        )
        key = "shareholder_wealth_" + price
        old = np.load(store / (key + ".npy"), mmap_mode="r")[day, name]
        control = np.float32(scale * q * raw)
        np.testing.assert_array_equal(control, old)
        corrected = np.float32(scale * raw)
        changes[key + "__indices"] = np.array([[day, name]])
        changes[key + "__values"] = np.array([corrected], np.float32)
        effects[key] = {"before": float(old), "after": float(corrected)}
    changes["m1_cotahist_return_consistent_mask__indices"] = np.array([[day, name]])
    changes["m1_cotahist_return_consistent_mask__values"] = np.array([True])
    np.savez_compressed(root / "deltas.npz", **changes)
    diagnostics = pl.DataFrame(
        evidence["rows"]["m1_cotahist_level_ratio"]
    ).with_columns(
        pl.col("trade_date").str.to_date(),
        pl.lit(True).alias("return_consistent"),
        pl.lit(False).alias("completed_action_boundary"),
    )
    diagnostics.write_parquet(root / "m1_diagnostic_replacement.parquet")
    # AERI has insufficient prefix return support, independently of the old
    # retrospective boundary. No target or new observed minute is fabricated.
    assert not np.load(store / "target_to_close_valid.npy", mmap_mode="r")[day, name]
    assert not np.load(store / "fast_present.npy", mmap_mode="r")[day, name]
    retained = {}
    for horizon in [1, 2, 3, 5, 10]:
        i = [1, 2, 3, 5, 10].index(horizon)
        earlier = day - horizon
        p = np.round(
            float(np.load(store / "raw_close.npy", mmap_mode="r")[earlier, name]), 2
        )
        expected = np.float32(5.71 / p - 1)
        actual = np.load(store / "target_shareholder_simple_return.npy", mmap_mode="r")[
            earlier, name, i
        ]
        valid = bool(
            np.load(store / "target_shareholder_valid.npy", mmap_mode="r")[
                earlier, name, i
            ]
        )
        if valid:
            np.testing.assert_array_equal(actual, expected)
        retained[str(horizon)] = {
            "date": str(dates[earlier]),
            "valid": valid,
            "simple_return": float(actual) if np.isfinite(actual) else None,
        }
    write_json_atomic(
        root / "wealth_qualification.json",
        {
            "evidence": binding(root / "manifest.json"),
            "deltas": binding(root / "deltas.npz"),
            "m1_diagnostic_replacement": binding(
                root / "m1_diagnostic_replacement.parquet"
            ),
            "wealth_effects": effects,
            "old_inferred_factor_control_exact": True,
            "new_q": 1,
            "new_cash": 0,
            "prefix_before_final_row_exact_by_sparse_scope": True,
            "primary_horizon_outcomes_retained": retained,
            "raw_price_simple_return": 5.71 / 8.31 - 1,
            "next_daily_feature_decision": "2025-01-02; outside this program's consumer axis, timestamp only",
            "through_2024_model_feature_or_target_changes": 0,
            "last_row_wealth_values_changed": 4,
            "diagnostic_cells_changed": 1,
            "issuer_pdf_visual_check": {
                "1312607": "Portuguese pp1-2: Dec9 debt covenant/payment negotiations, Dec30 debenture assembly; issuer/CNPJ/CVM/date verified",
                "1314227": "Portuguese p1 and English p2: Dec10 control-shareholder talks with Sinoma not advanced; no announced equity conversion/bonus",
            },
            "limits": "This updates the inherited wealth/diagnostic view to the accepted Round7 source disposition. It does not prove completeness of all issuer disclosures, identify a cause for the price move, or admit a hypothetical bonus. The policy account already uses q1/cash0/raw marks.",
            "executed_reproducer": binding(root / "wealth_executed.py"),
            "initial_attempts": [
                "date-column lookup failed before source retrieval",
                "renderer executable missing from uv PATH",
                "bundled Poppler has pdftoppm/pdfinfo but no pdftotext; reused downloaded bytes and extracted text with existing pypdf",
            ],
        },
    )
    print(json.dumps(effects), flush=True)


if __name__ == "__main__":
    main()
