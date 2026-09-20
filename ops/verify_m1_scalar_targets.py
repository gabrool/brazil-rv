"""Independent arithmetic oracle for the newly assembled to-close labels."""

import json
import shutil
from pathlib import Path
import numpy as np
import polars as pl
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.round5_derived import verified

p = Path(__file__).resolve().parents[1]
r = json.loads((p / "docs/v2_economic_data_scaling_run.json").read_text())
root = Path(r["root"]) / "m1_scalar_assembly"
shutil.copyfile(__file__, root / "target_oracle_executed.py")
m = bound_json(r["rename_m1_propagation"])
h = bound_json(r["rename_history_propagation"])
s = Path(m["parent"]["root"])
dates = np.load(s / "date_index.npy")
ids = np.load(s / "isin_index.npy")
active = np.load(verified(h["arrays"]["active"]), mmap_mode="r")
links = pl.read_parquet(verified(h["history_mapping"]))
count = 0
for rec in m["results"]:
    n = int(np.flatnonzero(ids == rec["isin"])[0])
    link = links.filter(pl.col("successor_index") == n).row(0, named=True)
    e = link["effective_index"]
    with np.load(root / "qualified_control" / str(dates[e]) / "raw_control.npz") as src:
        rows = src["date_indices"]
        raw = {k: src[k].copy() for k in src.files if k != "date_indices"}
    with np.load(verified(rec["raw_intraday"])) as inc:
        take = np.flatnonzero(
            (inc["date_indices"] >= max(e, link["known_index"]))
            & (inc["date_indices"] < e + 21)
        )
        local = inc["date_indices"][take] - rows[0]
        for k in [
            "return_consistent",
            "session_close",
            "session_close_valid",
            "realized_daily_vol",
            "fast_present",
        ]:
            raw[k][local, n] = inc[k][take, 1]
    out = np.load(root / "amendments" / (rec["isin"] + "_output.npz"))
    for j, day in enumerate(out["date_indices"]):
        i = int(day - rows[0])
        entry = raw["entry"][i].astype(float)
        close = raw["session_close"][i].astype(float)
        sigma = raw["realized_daily_vol"][i].astype(float)
        ok = (
            active[day]
            & raw["fast_present"][i]
            & raw["entry_valid"][i]
            & raw["return_consistent"][i]
            & raw["session_close_valid"][i]
            & np.isfinite(entry)
            & np.isfinite(close)
            & np.isfinite(sigma)
            & (entry > 0)
            & (close > 0)
            & (sigma > 0)
        )
        np.testing.assert_array_equal(ok, out["target_to_close_valid"][j])
        if not ok.any():
            continue
        returns = np.log(close[ok] / entry[ok])
        normalized = returns / (sigma[ok] * np.sqrt(60 / 405))
        normalized -= np.median(normalized)
        normalized = np.clip(normalized, -5, 5)
        unique, inverse, counts = np.unique(
            normalized, return_inverse=True, return_counts=True
        )
        rank = (np.cumsum(counts) - counts + (counts - 1) / 2)[inverse]
        expected = (
            (rank / (len(rank) - 1)).astype(np.float32)
            if len(rank) > 1
            else np.full(len(rank), 0.5, np.float32)
        )
        np.testing.assert_array_equal(expected, out["target_to_close"][j, ok])
        np.testing.assert_array_equal(
            normalized.astype(np.float32),
            out["target_to_close_normalized_residual"][j, ok],
        )
        np.testing.assert_array_equal(
            returns.astype(np.float32), out["target_to_close_raw_log_return"][j, ok]
        )
        count += int(ok.sum())
write_json_atomic(
    root / "target_oracle.json",
    {
        "source": binding(root / "amendments/manifest.json"),
        "valid_outcomes": count,
        "independent_return_residual_and_tie_rank_mismatches": 0,
        "executed_reproducer": binding(root / "target_oracle_executed.py"),
        "attribution_report": binding(root / "attribution_and_clock.json"),
        "attribution_source_resolution": {
            "path": str(p / "ops/audit_m1_scalar_attribution.py"),
            "sha256": binding(p / "ops/audit_m1_scalar_attribution.py")["sha256"],
            "change": "Formatting and replacing the minute lambda with the same named helper only; attribution computation was not rerun.",
        },
    },
)
print(json.dumps({"valid_outcomes": count, "mismatches": 0}))
